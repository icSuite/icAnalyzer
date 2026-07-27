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

class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.multi_channel = kwargs.pop('multi_channel', False)
        self.return_mask = kwargs.pop('return_mask', False)
        super(PartialConv2d, self).__init__(*args, **kwargs)
        self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
        self.slide_winsize = self.kernel_size[0] * self.kernel_size[1]
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        if mask_in is not None and mask_in.shape[1] > 1:
            mask_in = mask_in[:, 0:1, :, :] # Use one channel for mask
            
        with torch.no_grad():
            if self.weight_maskUpdater.type() != input.type():
                self.weight_maskUpdater = self.weight_maskUpdater.to(input)
            if mask_in is None:
                mask_in = torch.ones(input.size(0), 1, input.size(2), input.size(3)).to(input)

            update_mask = F.conv2d(mask_in, self.weight_maskUpdater, bias=None, 
                                   stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)
            self.mask_ratio = self.slide_winsize / (update_mask + 1e-8)
            self.update_mask = torch.clamp(update_mask, 0, 1)
            self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask_in))

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
        else:
            output = torch.mul(raw_out, self.mask_ratio)
            
        if self.return_mask:
            return output, self.update_mask
        else:
            return output

class VAE2D(nn.Module):
    def __init__(self, latent_dim=36):
        super(VAE2D, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        #self.enc_pconv1 = PartialConv2d(3, 32, kernel_size=3, stride=2, padding=1, return_mask=True)
        self.enc_pconv1 = RobustPartialConv2d(3, 32, kernel_size=3, stride=2, padding=1, return_mask=True)
        self.enc_bn1 = nn.BatchNorm2d(32)
        #self.enc_pconv2 = PartialConv2d(32, 64, kernel_size=3, stride=2, padding=1, return_mask=True)
        self.enc_pconv2 = RobustPartialConv2d(32, 64, kernel_size=3, stride=2, padding=1, return_mask=True)
        self.enc_bn2 = nn.BatchNorm2d(64)
        #self.enc_pconv3 = PartialConv2d(64, 128, kernel_size=3, stride=2, padding=1, return_mask=True)
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

class VAE2D_NoBN(nn.Module):
    def __init__(self, latent_dim=128): # Huge latent dim for capacity
        super(VAE2D_NoBN, self).__init__()
        self.latent_dim = latent_dim

        # Encoder (No BatchNorm!)
        self.enc_pconv1 = PartialConv2d(3, 32, kernel_size=3, stride=2, padding=1, return_mask=True)
        self.enc_pconv2 = PartialConv2d(32, 64, kernel_size=3, stride=2, padding=1, return_mask=True)
        self.enc_pconv3 = PartialConv2d(64, 128, kernel_size=3, stride=2, padding=1, return_mask=True)
        
        self.flatten_dim = 128 * 5 * 5 
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder (No BatchNorm!)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(32, 2, kernel_size=3, stride=2, padding=1, output_padding=1)

    def encode(self, x, mask):
        x1, m1 = self.enc_pconv1(x, mask_in=mask)
        x1 = F.relu(x1) # Removed BN
        x2, m2 = self.enc_pconv2(x1, mask_in=m1)
        x2 = F.relu(x2) # Removed BN
        x3, m3 = self.enc_pconv3(x2, mask_in=m2)
        x3 = F.relu(x3) # Removed BN
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
        x = F.relu(self.dec_conv1(x)) # Removed BN
        x = F.relu(self.dec_conv2(x)) # Removed BN
        x = self.dec_conv3(x) 
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

#%% 4. DIAGNOSTIC TESTS

def run_capacity_test():
    """Phase 1: Overfit Check on a small batch."""
    print("\n--- PHASE 1: CAPACITY TEST ---")
    
    # 1. Load Data (Strict filter, only perfect data)
    data, mask, _ = load_data(max_sparsity=0.05, subset_size=5)
    
    # Take just one batch of 16 images
    dataset = TensorDataset(data[:16], mask[:16])
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    model = VAE2D(latent_dim=128).to(CONFIG['DEVICE']) # Big latent dim to ensure capacity
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Monitor 'min' loss. If it doesn't improve for 50 epochs, cut LR by 10x.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=50, verbose=True
    )
    
    losses = []
    lrs = []
    
    print("Overfitting PartialConv VAE...")
    # 2000 Epochs as per your finding
    pbar = tqdm(range(20000)) 
    
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
    
            
    # Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Capacity Loss (Target: -> 0)")
    plt.yscale('log')
    
    # Show Reconstruction
    with torch.no_grad():
        recon, _, _ = model(x_batch, m_batch)
        
    plt.subplot(1, 2, 2)
    # Compare first image Hall
    in_img = x_batch[0, 0].cpu().numpy()
    out_img = recon[0, 0].cpu().numpy()
    combined = np.hstack([in_img, out_img])
    plt.imshow(combined, cmap='plasma', origin='lower')
    plt.title("Input vs Output (Hall)")
    plt.show()

def run_quantitative_lr_sweep():
    print("\n--- PHASE 2: QUANTITATIVE LR SWEEP ---")
    
    # 1. Load Representative Subset (50 orbits)
    data, mask, _ = load_data(max_sparsity=0.4, subset_size=50)
    dataset = TensorDataset(data, mask)
    
    lrs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    results = {}
    
    plt.figure(figsize=(10, 6))
    
    for lr in lrs:
        print(f"Testing LR: {lr}...")
        model = VAE2D(latent_dim=36).to(CONFIG['DEVICE']) 
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        losses = []
        
        # Run for 50 Epochs
        for epoch in range(50):
            epoch_loss = 0
            for x, m in dataloader:
                x, m = x.to(CONFIG['DEVICE']), m.to(CONFIG['DEVICE'])
                recon, mu, logvar = model(x, m)
                loss, _, _ = loss_function(recon, x, m, mu, logvar, beta=1.0)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(dataloader))

        # --- CALCULATE METRICS ---
        # 1. Convergence (AUC) - lower is better
        auc = np.sum(losses)
        # 2. Final Performance (Last 10 avg)
        final_loss = np.mean(losses[-10:])
        # 3. Stability (Last 10 std dev)
        stability = np.std(losses[-10:])
        
        results[lr] = {'losses': losses, 'auc': auc, 'final': final_loss, 'stab': stability}
        
        plt.plot(losses, label=f"LR={lr:.0e} (AUC={auc:.1f})")

    plt.yscale('log')
    plt.legend()
    plt.title("Learning Rate Optimization")
    plt.show()
    
    # --- GENERATE LATEX TABLE ---
    print("\n--- LATEX TABLE GEN ---")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lccc}")
    print("\\hline")
    print("Learning Rate & Convergence (AUC) $\\downarrow$ & Final Loss $\\downarrow$ & Stability ($\\sigma$) $\\downarrow$ \\\\")
    print("\\hline")
    for lr in lrs:
        r = results[lr]
        # Format scientific notation for LaTeX
        lr_str = f"{lr:.0e}".replace("e-0", "e-")
        print(f"${lr_str}$ & {r['auc']:.2f} & {r['final']:.4f} & {r['stab']:.5f} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Quantitative comparison of learning rates. AUC represents the area under the loss curve (convergence speed).}")
    print("\\label{tab:lr_sweep}")
    print("\\end{table}")

def run_lr_sweep():
    print("\n--- PHASE 2: LEARNING RATE SWEEP ---")
    
    # 1. Load a "Representative" Subset
    # Not just 5 orbits (too easy), but not all 500 (too slow).
    # 50 orbits gives a good approximation of the full dataset's variance.
    data, mask, _ = load_data(max_sparsity=0.4, subset_size=50)
    
    # Standard Dataset Setup
    dataset = TensorDataset(data, mask)
    
    # Test Candidates
    lrs = [1e-3, 1e-4, 1e-5]
    colors = ['red', 'green', 'blue']
    
    plt.figure(figsize=(10, 6))
    
    for lr, col in zip(lrs, colors):
        print(f"Testing LR: {lr}...")
        
        # Fresh Model & Optimizer for each run
        model = VAE2D(latent_dim=36).to(CONFIG['DEVICE']) 
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Safe Batch Size
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        losses = []
        
        # Short Sprint (50 Epochs)
        # We don't need to converge, just see the slope and stability.
        for epoch in range(50):
            epoch_loss = 0
            for x, m in dataloader:
                x, m = x.to(CONFIG['DEVICE']), m.to(CONFIG['DEVICE'])
                
                recon, mu, logvar = model(x, m)
                
                # Standard Loss (Beta=1.0 for a fair VAE test)
                loss, _, _ = loss_function(recon, x, m, mu, logvar, beta=1.0)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            # Print first few to ensure it's starting
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: {avg_loss:.4f}")

        # Plot this curve
        plt.plot(losses, color=col, label=f"LR={lr}", linewidth=2)
    
    plt.title("Learning Rate Drag Race (50 Epochs)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Log Scale)")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()

def run_rate_distortion_analysis():
    print("\n--- PHASE 3: QUANTITATIVE BETA SWEEP (Rate-Distortion) ---")
    
    # 1. Load Data
    data, mask, _ = load_data(max_sparsity=0.4, subset_size=50) 
    dataset = TensorDataset(data, mask)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Finer grid, focusing on the transition zone
    betas = [0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    
    results = {'beta': [], 'recon': [], 'kl': [], 'total': []}
    
    for beta in betas:
        print(f"Testing Beta={beta}...")
        model = VAE2D(latent_dim=36).to(CONFIG['DEVICE'])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Fast training (30 epochs is enough to settle the loss terms)
        for epoch in range(30):
            for x, m in dataloader:
                x, m = x.to(CONFIG['DEVICE']), m.to(CONFIG['DEVICE'])
                recon, mu, logvar = model(x, m)
                loss, recon_l, kld_l = loss_function(recon, x, m, mu, logvar, beta=beta)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # EVALUATE (Average over last epoch)
        # We need separate Recon and KL values
        model.eval()
        running_recon = 0
        running_kl = 0
        with torch.no_grad():
            for x, m in dataloader:
                x, m = x.to(CONFIG['DEVICE']), m.to(CONFIG['DEVICE'])
                recon, mu, logvar = model(x, m)
                _, r_l, k_l = loss_function(recon, x, m, mu, logvar, beta=beta)
                running_recon += r_l.item()
                running_kl += k_l.item()
        
        avg_recon = running_recon / len(dataloader)
        avg_kl = running_kl / len(dataloader)
        
        results['beta'].append(beta)
        results['recon'].append(avg_recon)
        results['kl'].append(avg_kl)
        print(f"  -> Recon: {avg_recon:.1f} | KL: {avg_kl:.1f}")

    # --- PLOT 1: The Dual-Axis Trade-off ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Beta (Log Scale)')
    ax1.set_ylabel('Reconstruction Error (Distortion)', color=color)
    ax1.plot(results['beta'], results['recon'], color=color, marker='o', label='Reconstruction')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('symlog', linthresh=0.001) # Handle 0.0 correctly

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('KL Divergence (Rate)', color=color)  
    ax2.plot(results['beta'], results['kl'], color=color, marker='s', linestyle='--', label='KL Divergence')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("The Rate-Distortion Curve (Selecting Beta)")
    plt.grid(True, which='both', alpha=0.3)
    plt.show()

    # --- PLOT 2: The Elbow Curve (Recon vs KL directly) ---
    # This is the standard "Pareto Frontier" plot
    plt.figure(figsize=(8, 6))
    plt.plot(results['kl'], results['recon'], marker='o', linestyle='-')
    for i, txt in enumerate(results['beta']):
        plt.annotate(f"$\\beta={txt}$", (results['kl'][i], results['recon'][i]), xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel("Information Rate (KL Divergence)")
    plt.ylabel("Distortion (Reconstruction Error)")
    plt.title("Pareto Frontier: Ideally closer to (0,0)")
    plt.grid(True)
    # Invert X axis because usually we want low KL, but higher beta reduces KL
    # Actually, standard is KL on X. 
    # High Beta = Low KL, High Error (Left, Top)
    # Low Beta = High KL, Low Error (Right, Bottom)
    plt.show()

def run_beta_sweep_final():
    print("\n--- PHASE 3: BETA SWEEP (Final Settings) ---")
    
    # 1. Load Data (50 orbits, relaxed sparsity)
    data, mask, stats = load_data(max_sparsity=0.4, subset_size=50) 
    
    dataset = TensorDataset(data, mask)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    betas = [0.0, 0.1, 1.0]
    
    # Create plot grid: 3 Rows (Betas) x 3 Cols (Loss, Recon, Sample)
    fig, axs = plt.subplots(len(betas), 3, figsize=(15, 12))
    
    for i, beta in enumerate(betas):
        print(f"\nTraining Beta={beta}...")
        model = VAE2D(latent_dim=36).to(CONFIG['DEVICE'])
        
        # Use the WINNER from your sweep
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Scheduler for fine-tuning
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10, verbose=True
        )
        
        losses = []
        
        # Train for 100 epochs
        pbar = tqdm(range(100))
        for epoch in pbar:
            epoch_loss = 0
            for x_batch, m_batch in dataloader:
                x_batch = x_batch.to(CONFIG['DEVICE'])
                m_batch = m_batch.to(CONFIG['DEVICE'])
                
                recon, mu, logvar = model(x_batch, m_batch)
                
                loss, recon_l, kld_l = loss_function(recon, x_batch, m_batch, mu, logvar, beta=beta)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            pbar.set_description(f"Loss: {avg_loss:.4f}")
            
            scheduler.step(avg_loss)
            
        # --- DIAGNOSTICS ---
        # 1. Loss Curve
        axs[i, 0].plot(losses)
        axs[i, 0].set_title(f"Beta={beta} | Loss Curve")
        axs[i, 0].set_yscale('log')
        axs[i, 0].set_ylabel("Total Loss")
        
        # 2. Reconstruction (Validation)
        with torch.no_grad():
            x_sample = data[0:1].to(CONFIG['DEVICE'])
            m_sample = mask[0:1].to(CONFIG['DEVICE'])
            rec_sample, _, _ = model(x_sample, m_sample)
            
            orig = x_sample[0, 0].cpu().numpy()
            rec = rec_sample[0, 0].cpu().numpy() * m_sample[0, 0].cpu().numpy()
            
            # Combine Side-by-Side
            combined = np.hstack([orig, rec])
            axs[i, 1].imshow(combined, origin='lower', cmap='plasma')
            axs[i, 1].set_title(f"Input vs Recon (Hall)")
            
        # 3. Latent Sampling (Generation)
        # We sample z ~ N(0, 1) to see if it generates a valid "Ghost Aurora"
        with torch.no_grad():
            z_rand = torch.randn(1, 36).to(CONFIG['DEVICE'])
            gen_out = model.decode(z_rand)
            gen_img = gen_out[0, 0].cpu().numpy()
            axs[i, 2].imshow(gen_img, origin='lower', cmap='plasma')
            axs[i, 2].set_title(f"Random Sample z~N(0,1)")

    plt.tight_layout()
    plt.show()

def run_beta_sweep():
    """Phase 3: Beta Trade-off (Recon vs Smoothness)."""
    print("\n--- PHASE 3: BETA MANIFOLD CHECK ---")
    
    data, mask, stats = load_data(max_sparsity=0.3, subset_size=20)
    dataset = TensorDataset(data, mask)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    betas = [0.1, 1.0, 5.0]
    
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    # Rows: Betas
    # Col 1: Reconstruction
    # Col 2: Random Sample (Generation)
    # Col 3: Latent Scatter
    
    for i, beta in enumerate(betas):
        print(f"Training Beta={beta}...")
        model = VAE2D(latent_dim=36).to(CONFIG['DEVICE'])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Train for 20 epochs
        for epoch in range(20):
            for x, m in dataloader:
                x, m = x.to(CONFIG['DEVICE']), m.to(CONFIG['DEVICE'])
                optimizer.zero_grad()
                recon, mu, logvar = model(x, m)
                loss, _, _ = loss_function(recon, x, m, mu, logvar, beta=beta)
                loss.backward()
                optimizer.step()
                
        # 1. Reconstruction Quality
        with torch.no_grad():
            x_sample = x[0:1] # Take first from last batch
            m_sample = m[0:1]
            rec_sample, _, _ = model(x_sample, m_sample)
            
            orig = x_sample[0, 0].cpu().numpy()
            rec = rec_sample[0, 0].cpu().numpy()
            axs[i, 0].imshow(np.hstack([orig, rec]), origin='lower', cmap='plasma')
            axs[i, 0].set_title(f"Beta={beta} | In vs Rec")
            
        # 2. Random Generation (Smoothness)
        with torch.no_grad():
            z_rand = torch.randn(1, 36).to(CONFIG['DEVICE'])
            gen_out = model.decode(z_rand)
            gen_img = gen_out[0, 0].cpu().numpy()
            axs[i, 1].imshow(gen_img, origin='lower', cmap='plasma')
            axs[i, 1].set_title(f"Beta={beta} | Random Sample")
            
        # 3. Latent Space (PCA)
        # Collect latents
        mus = []
        for x, m in dataloader:
             x, m = x.to(CONFIG['DEVICE']), m.to(CONFIG['DEVICE'])
             with torch.no_grad():
                 _, mu_batch, _ = model(x, m)
                 mus.append(mu_batch.cpu().numpy())
        mus = np.concatenate(mus, axis=0)
        
        # Simple PCA projection
        cov = np.cov(mus.T)
        vals, vecs = np.linalg.eigh(cov)
        # Project onto top 2
        proj = mus @ vecs[:, -2:]
        
        axs[i, 2].scatter(proj[:, 0], proj[:, 1], s=2, alpha=0.5)
        axs[i, 2].set_title(f"Beta={beta} | Latent Space")
        
    plt.tight_layout()
    plt.show()

def run_sparsity_test():
    """Phase 4: Robustness to Sparse Training Data."""
    print("\n--- PHASE 4: SPARSITY ROBUSTNESS ---")
    
    # Create a "Clean Test Set" (f < 0.1)
    clean_data, clean_mask, _ = load_data(max_sparsity=0.1, subset_size=10)
    test_loader = DataLoader(TensorDataset(clean_data, clean_mask), batch_size=16)
    
    # Train on increasingly messy data
    sparsity_levels = [0.1, 0.4, 0.7] 
    val_losses = []
    
    for sp in sparsity_levels:
        print(f"Training with Max Missing Fraction = {sp}...")
        train_data, train_mask, _ = load_data(max_sparsity=sp, subset_size=20)
        train_loader = DataLoader(TensorDataset(train_data, train_mask), batch_size=16, shuffle=True)
        
        model = VAE2D(latent_dim=36).to(CONFIG['DEVICE'])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Train
        for epoch in range(15):
            model.train()
            for x, m in train_loader:
                x, m = x.to(CONFIG['DEVICE']), m.to(CONFIG['DEVICE'])
                optimizer.zero_grad()
                recon, mu, logvar = model(x, m)
                loss, _, _ = loss_function(recon, x, m, mu, logvar, beta=1.0)
                loss.backward()
                optimizer.step()
        
        # Evaluate on Clean Test Set
        model.eval()
        total_recon_loss = 0
        with torch.no_grad():
            for x, m in test_loader:
                x, m = x.to(CONFIG['DEVICE']), m.to(CONFIG['DEVICE'])
                recon, mu, logvar = model(x, m)
                _, r_loss, _ = loss_function(recon, x, m, mu, logvar, beta=1.0)
                total_recon_loss += r_loss.item()
        
        avg_loss = total_recon_loss / len(test_loader)
        val_losses.append(avg_loss)
        print(f"Sparsity {sp} -> Test Loss: {avg_loss:.4f}")
        
    plt.figure()
    plt.plot(sparsity_levels, val_losses, marker='o')
    plt.xlabel("Max Missing Fraction allowed in Training")
    plt.ylabel("Reconstruction Error on CLEAN Data")
    plt.title("Does adding sparse data hurt clean performance?")
    plt.grid(True)
    plt.show()

#%% 5. MAIN EXECUTION

if __name__ == "__main__":
    # SELECT TEST HERE: 'capacity', 'lr', 'beta', 'sparsity'
    TEST_TO_RUN = 'beta' 
    
    if TEST_TO_RUN == 'capacity':
        run_capacity_test()
    elif TEST_TO_RUN == 'lr':
        #run_lr_sweep()
        run_quantitative_lr_sweep()
    elif TEST_TO_RUN == 'beta':
        #run_beta_sweep()
        #run_beta_sweep_final()
        run_rate_distortion_analysis()
    elif TEST_TO_RUN == 'sparsity':
        run_sparsity_test()