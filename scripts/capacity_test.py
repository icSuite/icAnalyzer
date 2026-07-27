#%% 1. IMPORTS & CONFIGURATION
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join as pjoin
from tqdm import tqdm
import copy

# Try importing custom library
try:
    from icreader import ConductanceImage
except ImportError:
    print("Warning: 'icreader' not found. Ensure data loading works.")

CONFIG = {
    'IMG_HEIGHT': 36,
    'IMG_WIDTH': 36,
    'LATENT_DIM': 36,
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'SEED': 42
}

# Paths (Adjust as needed)
PATHS = {
    'data_in': '/home/bing/Dropbox/work/data/IMAGE_FUV/conductance',
    'segments': '/home/bing/Dropbox/work/code/repos/icAnalyzer/data/good_orbit_segments.csv'
}

np.random.seed(CONFIG['SEED'])
torch.manual_seed(CONFIG['SEED'])

#%% 2. MODEL DEFINITION (PartialConv VAE)

class RobustPartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.return_mask = kwargs.pop('return_mask', False)
        self.multi_channel = kwargs.pop('multi_channel', False)
        super(RobustPartialConv2d, self).__init__(*args, **kwargs)

        # Kernel for updating the mask (Count valid pixels)
        self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
        self.slide_winsize = self.kernel_size[0] * self.kernel_size[1]
        
        # Initialize weights with Kaiming Uniform (Better for Relu)
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, input, mask_in=None):
        # input: (B, C, H, W)
        # mask_in: (B, 1, H, W) or None
        
        if mask_in is not None:
            # Ensure mask is 1 channel
            if mask_in.shape[1] > 1: mask_in = mask_in[:, 0:1, :, :]
            
            with torch.no_grad():
                # 1. Update Mask: Convolution with all-ones kernel counts valid pixels
                # padding=0 because we want to shrink the mask if edges are invalid? 
                # No, use same padding as main conv to keep dimensions
                self.weight_maskUpdater = self.weight_maskUpdater.to(input.device)
                
                update_mask = F.conv2d(mask_in, self.weight_maskUpdater, bias=None, 
                                       stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

                # 2. Calculate Scaling Factor
                # If window has 9 pixels but only 3 are valid, we multiply result by 9/3 = 3
                # eps prevents division by zero
                mask_ratio = self.slide_winsize / (update_mask + 1e-6)
                
                # If no pixels are valid, ratio is useless. Clamp it.
                # update_mask == 0 implies ratio is huge. 
                # We essentially want to Zero out the output where mask is 0.
                self.update_mask = torch.clamp(update_mask, 0, 1) # Binary new mask
                mask_ratio = mask_ratio * self.update_mask # Zeros out invalid regions

        # 3. Standard Convolution on Masked Input
        # (Input * Mask) ensures we don't conv over garbage/NaNs
        raw_out = super(RobustPartialConv2d, self).forward(input * mask_in if mask_in is not None else input)

        if self.bias is not None:
            # bias is added to everything, but we only want it where mask is valid
            # And we need to scale the (conv + bias)? No, usually just scale conv.
            # Nvidia paper: output = (W^T . (X*M) * ratio) + b
            output = (raw_out * mask_ratio) + self.bias.view(1, -1, 1, 1)
            # Mask the output bias too
            output = output * self.update_mask
        else:
            output = raw_out * mask_ratio

        if self.return_mask:
            return output, self.update_mask
        else:
            return output

class VAE2D(nn.Module):
    def __init__(self, latent_dim=36):
        super(VAE2D, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.enc_pconv1 = RobustPartialConv2d(3, 32, kernel_size=3, stride=2, padding=1, return_mask=True)
        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_pconv2 = RobustPartialConv2d(32, 64, kernel_size=3, stride=2, padding=1, return_mask=True)
        self.enc_bn2 = nn.BatchNorm2d(64)
        self.enc_pconv3 = RobustPartialConv2d(64, 128, kernel_size=3, stride=2, padding=1, return_mask=True)
        self.enc_bn3 = nn.BatchNorm2d(128)
        
        self.flatten_dim = 128 * 5 * 5 
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.dec_bn1 = nn.BatchNorm2d(64)
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn2 = nn.BatchNorm2d(32)
        self.dec_conv3 = nn.ConvTranspose2d(32, 2, kernel_size=3, stride=2, padding=1, output_padding=1)

    def encode(self, x, mask):
        x1, m1 = self.enc_pconv1(x, mask_in=mask)
        x1 = F.relu(self.enc_bn1(x1))
        x2, m2 = self.enc_pconv2(x1, mask_in=m1)
        x2 = F.relu(self.enc_bn2(x2))
        x3, m3 = self.enc_pconv3(x2, mask_in=m2)
        x3 = F.relu(self.enc_bn3(x3))
        x_flat = x3.view(x3.size(0), -1)
        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(x.size(0), 128, 5, 5) 
        x = F.relu(self.dec_bn1(self.dec_conv1(x)))
        x = F.relu(self.dec_bn2(self.dec_conv2(x)))
        x = self.dec_conv3(x) # Linear output (Log-space or raw)
        return x

    def forward(self, x, mask):
        mu, logvar = self.encode(x, mask)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def loss_function(recon, x, mask, mu, logvar, beta=1.0):
    # x: [B, 3, H, W] -> (H, P, W)
    target = x[:, :2, :, :]
    weights = x[:, 2:3, :, :] 
    
    # Mask * HuberWeight
    final_weight = mask * weights
    
    # Weighted MSE
    recon_loss = torch.sum(final_weight * (recon - target)**2) / (final_weight.sum() + 1e-8)
    # Scale up loss to make it roughly O(1) ~ O(10)
    recon_loss *= (36*36*2) 

    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalize KLD by batch size
    kld_loss /= x.size(0)

    return recon_loss + beta * kld_loss, recon_loss, kld_loss

#%% 3. DATA LOADING UTILS

def load_data(max_sparsity=0.5, subset_size=None):
    """Loads data with specific sparsity threshold."""
    print(f"Loading data with max_sparsity={max_sparsity}...")
    gos = pd.read_csv(PATHS['segments'], header=None)
    gos.columns = ['orbit', 'shit_start', 'shit_stop', 'good_start', 'good_stop']
    gos = gos.loc[gos['orbit'] < 503, :]

    H_list, P_list, W_list = [], [], []
    
    # Limit to first 20 orbits for quick testing if subset_size is set, else all
    orbit_limit = 20 if subset_size else 9999
    
    count = 0
    for orbit in tqdm(gos['orbit']):
        if count >= orbit_limit: break
        try:
            cI = ConductanceImage(pjoin(PATHS['data_in'], f'or_{str(orbit).zfill(4)}.nc'))
            # Get valid frames
            h_orb, p_orb, w_orb = cI.H, cI.P, cI.w
            
            # Filter by sparsity
            h, w = CONFIG['IMG_HEIGHT'], CONFIG['IMG_WIDTH']
            missing_frac = np.sum((np.isnan(h_orb) | np.isnan(p_orb)), axis=(1,2)) / (h*w)
            
            # Keep frames that are clean enough
            valid_idx = missing_frac < max_sparsity
            
            if np.sum(valid_idx) > 0:
                H_list.append(h_orb[valid_idx])
                P_list.append(p_orb[valid_idx])
                W_list.append(w_orb[valid_idx])
                count += 1
        except: continue

    H = np.concatenate(H_list, axis=0)
    P = np.concatenate(P_list, axis=0)
    W = np.concatenate(W_list, axis=0)
    
    # Normalize
    H_mu, H_sig = np.nanmean(H), np.nanstd(H)
    P_mu, P_sig = np.nanmean(P), np.nanstd(P)
    
    H_norm = (H - H_mu) / H_sig
    P_norm = (P - P_mu) / P_sig
    
    # Replace NaNs with 0 for Tensor conversion (Mask handles logic)
    X_np = np.stack([np.nan_to_num(H_norm, nan=0.0), 
                     np.nan_to_num(P_norm, nan=0.0),
                     np.nan_to_num(W, nan=0.0)], axis=1)
    
    # Create Mask
    mask_np = (~(np.isnan(H) | np.isnan(P))).astype(np.float32)
    mask_np = np.expand_dims(mask_np, 1) # (N, 1, 36, 36)
    
    return torch.from_numpy(X_np).float(), torch.from_numpy(mask_np).float(), (H_mu, H_sig, P_mu, P_sig)

#%% 5. MAIN EXECUTION

def run_partial_conv_capacity_test_with_scheduler():
    print("\n--- PARTIAL CONV CAPACITY TEST (With Scheduler) ---")
    
    dat_idx = np.arange(16).astype(int)*100 + 32
    # 1. Load Data (Small subset to memorize)
    data, mask, _ = load_data(max_sparsity=0.5, subset_size=50)
    
    fig, axs = plt.subplots(4, 4, figsize=(10,10))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(data[dat_idx[i], 0].cpu().numpy(), cmap='plasma', origin='lower')
        ax.set_axis_off()
    plt.suptitle('Input Hall')
    
    fig, axs = plt.subplots(4, 4, figsize=(10,10))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(data[dat_idx[i], 1].cpu().numpy(), cmap='plasma', origin='lower')        
        ax.set_axis_off()
    plt.suptitle('Input Pedersen')
    
    # Take 16 images
    dataset = TensorDataset(data[dat_idx], mask[dat_idx])
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    loss_ = []
    lrs_ = []
    out_img_H = []
    out_img_P = []
    ltspace_ = [1024, 512, 256, 128, 64, 32, 16, 8, 4]
    
    
    for ltspace in ltspace_:
    
        # 2. The Model: VAE2D with RobustPartialConv2d
        model = VAE2D(latent_dim=ltspace).to(CONFIG['DEVICE']) 
        
        # 3. Optimizer & Scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Monitor 'min' loss. If it doesn't improve for 50 epochs, cut LR by 10x.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=25, verbose=True
        )
        
        losses = []
        lrs = []
        
        print("Overfitting PartialConv VAE...")
        # 2000 Epochs as per your finding
        pbar = tqdm(range(2000)) 
        
        for epoch in pbar:
            epoch_loss = 0
            for x_batch, m_batch in dataloader:
                x_batch = x_batch.to(CONFIG['DEVICE'])
                m_batch = m_batch.to(CONFIG['DEVICE'])
                
                # Forward
                recon, mu, logvar = model(x_batch, m_batch)
                
                # Loss
                loss, recon_l, _ = loss_function(recon, x_batch, m_batch, mu, logvar, beta=0.0) # Beta=0 for pure capacity check
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
    
            # Step Scheduler based on the loss
            scheduler.step(epoch_loss)
            
            losses.append(epoch_loss)
            lrs.append(optimizer.param_groups[0]['lr'])
            pbar.set_description(f"Loss: {epoch_loss:.4f} | LR: {lrs[-1]:.1e}")
    
        print(f"Final Loss: {losses[-1]:.6f}")
        
        loss_.append(losses)
        lrs_.append(lrs)
    
        # Visual Check
        with torch.no_grad():
            recon, _, _ = model(data[dat_idx].to(CONFIG['DEVICE']), mask[dat_idx].to(CONFIG['DEVICE']))
        
        
        out_img_H_ = recon[0, 0].cpu().numpy()
        out_img_P_ = recon[0, 1].cpu().numpy()
        
        # Mask output for fair visual comparison (we don't care about the void)

        out_img_H.append(out_img_H_ * mask[dat_idx[0], 0].cpu().numpy())
        out_img_P.append(out_img_P_ * mask[dat_idx[0], 0].cpu().numpy())
    
    in_img_H = data[dat_idx[0], 0].cpu().numpy()
    in_img_P = data[dat_idx[0], 1].cpu().numpy()
    
    rmse_H = [np.nansum((in_img_H - x)**2) for x in out_img_H]
    rmse_P = [np.nansum((in_img_P - x)**2) for x in out_img_P]
    
    fig = plt.figure(figsize=(15, 9))
    plt.semilogy(ltspace_, rmse_H, '.-', label='Hall')
    plt.semilogy(ltspace_, rmse_P, '.-', label='Pedersen')
    plt.xlabel('Laten Space')
    plt.ylabel('RMSE')
    plt.legend()
    
    fig = plt.figure(figsize=(15, 9))
    for ltspace, loss in zip(ltspace_, loss_):
        plt.semilogy(loss, label=f'{ltspace}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    vmin = 0
    vmax = np.nanmax(in_img_H)
    out_img_H.insert(0, in_img_H)
    fig, axs = plt.subplots(2, 5, figsize=(15, 9))
    for ax, img in zip(axs.flatten(), out_img_H):
        ax.imshow(img, cmap='plasma', origin='lower', vmin=vmin, vmax=vmax)
    plt.suptitle('Hall')
    
    vmax = np.nanmax(in_img_P)
    out_img_P.insert(0, in_img_P)
    fig, axs = plt.subplots(2, 5, figsize=(15, 9))
    for ax, img in zip(axs.flatten(), out_img_P):
        ax.imshow(img, cmap='plasma', origin='lower', vmin=vmin, vmax=vmax)
    plt.suptitle('Pedersen')

    return lrs_, loss_

if __name__ == "__main__":
    lrs, loss = run_partial_conv_capacity_test_with_scheduler()
