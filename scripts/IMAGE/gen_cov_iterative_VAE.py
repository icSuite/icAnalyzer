import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join as pjoin
from tqdm import tqdm
import math
import os

# Import your helper
os.chdir('/home/bing/Dropbox/work/code/repos/icAnalyzer/scripts/IMAGE')
import helpers 

# --- CONFIGURATION ---
CONFIG = {
    'IMG_HEIGHT': 36,
    'IMG_WIDTH': 36,
    'LATENT_DIM': 36,         # Bottleneck size
    'BATCH_SIZE': 8,         # Stability optimized
    'LR': 1e-4,               # Learning Rate (Optimized)
    'BETA': 1,              # Physics/Recon Trade-off (Optimized)
    'EPOCHS': 100,
    'SPARSITY_THRESHOLD': 0.4,
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'SEED': 42
}

PATHS = {
    'data_in': '/home/bing/Dropbox/work/data/IMAGE_FUV/conductance',
    'segments': '/home/bing/Dropbox/work/code/repos/icAnalyzer/data/good_orbit_segments.csv',
    'save_model': 'vae_aurora_linear.pth'
}

np.random.seed(CONFIG['SEED'])
torch.manual_seed(CONFIG['SEED'])

# --- 2. MODEL ARCHITECTURE (Robust PartialConv) ---

class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.multi_channel = kwargs.pop('multi_channel', False)
        self.return_mask = kwargs.pop('return_mask', False)
        super(PartialConv2d, self).__init__(*args, **kwargs)

        self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
        self.slide_winsize = self.kernel_size[0] * self.kernel_size[1]
        self.last_size = (0, 0, 0, 0)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        if mask_in is not None and mask_in.shape[1] > 1:
            mask_in = mask_in[:, 0:1, :, :]
            
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
        self.enc1 = PartialConv2d(3, 32, 3, 2, 1, return_mask=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.enc2 = PartialConv2d(32, 64, 3, 2, 1, return_mask=True)
        self.bn2 = nn.BatchNorm2d(64)
        self.enc3 = PartialConv2d(64, 128, 3, 2, 1, return_mask=True)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.flatten_dim = 128 * 5 * 5 
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        self.dec1 = nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=0)
        self.bn4 = nn.BatchNorm2d(64)
        self.dec2 = nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.dec3 = nn.ConvTranspose2d(32, 2, 3, 2, 1, output_padding=1)

    def encode(self, x, mask):
        x1, m1 = self.enc1(x, mask_in=mask); x1 = F.relu(self.bn1(x1))
        x2, m2 = self.enc2(x1, mask_in=m1); x2 = F.relu(self.bn2(x2))
        x3, m3 = self.enc3(x2, mask_in=m2); x3 = F.relu(self.bn3(x3))
        x_flat = x3.view(x3.size(0), -1)
        return self.fc_mu(x_flat), self.fc_logvar(x_flat)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        x = self.fc_decode(z).view(-1, 128, 5, 5)
        x = F.leaky_relu(self.bn4(self.dec1(x)))
        x = F.leaky_relu(self.bn5(self.dec2(x)))
        return self.dec3(x)

    def forward(self, x, mask):
        mu, logvar = self.encode(x, mask)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon, x, mask, mu, logvar, beta):
    target = x[:, :2]
    weights = x[:, 2:3] * mask 
    
    # Weighted MSE
    recon_loss = torch.sum(weights * (recon - target)**2) / (weights.sum() + 1e-8) * (36*36*2)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
    return recon_loss + beta * kld_loss, recon_loss, kld_loss

# --- 3. ANALYSIS CLASS ---

class AuroralAnalyzer:
    def __init__(self, model, stats):
        self.model = model
        self.stats = stats
        self.H_flat = CONFIG['IMG_HEIGHT'] * CONFIG['IMG_WIDTH']
        
    def estimate_linear_covariance(self, n_samples=5000):
        print(f"Estimating LINEAR Covariance from {n_samples} samples...")
        self.model.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, CONFIG['LATENT_DIM']).to(CONFIG['DEVICE'])
            out = self.model.decode(z).cpu().numpy()
            
            # Denormalize to Physics Units (Linear Mho)
            h_phys = (out[:,0] * self.stats['H_sig']) + self.stats['H_mu']
            p_phys = (out[:,1] * self.stats['P_sig']) + self.stats['P_mu']
            
            # Enforce Non-Negativity (The "Clipping" Strategy)
            h_phys = np.maximum(h_phys, 0)
            p_phys = np.maximum(p_phys, 0)
            
            # Stack State Vector [H, P]
            X = np.hstack([h_phys.reshape(n_samples, -1), p_phys.reshape(n_samples, -1)])
            
            self.mean_vec = np.mean(X, axis=0)
            self.cov_matrix = np.cov(X, rowvar=False)
            self.X_samples = X

    def plot_covariance_matrix(self):
        
        plt.figure(figsize=(10, 8))
        cov_vis = self.cov_matrix[::4, ::4]
        
        plt.imshow(np.abs(cov_vis), cmap='inferno', origin='upper')
        plt.colorbar(label='Covariance (Mho^2)')
        
        mid = cov_vis.shape[0] // 2
        plt.axhline(mid, color='w', linestyle='--'); plt.axvline(mid, color='w', linestyle='--')
        plt.text(mid/2, mid/2, 'Hall', color='w', ha='center', fontweight='bold')
        plt.text(3*mid/2, 3*mid/2, 'Pedersen', color='w', ha='center', fontweight='bold')
        plt.title("VAE Covariance (Linear Space)")
        plt.show()

    def plot_teleconnections(self, probe_idx=None):
        if probe_idx is None: probe_idx = (18 * 36) + 18 
        d = np.sqrt(np.diag(self.cov_matrix))
        d[d==0] = 1e-9 
        corr_matrix = self.cov_matrix / np.outer(d, d)
        
        corr_row = corr_matrix[probe_idx, :]
        corr_h = corr_row[:self.H_flat].reshape(36, 36)
        corr_p = corr_row[self.H_flat:].reshape(36, 36)
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        py, px = divmod(probe_idx, 36)
        
        im1 = axs[0].imshow(corr_h, cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
        axs[0].scatter(px, py, c='lime', marker='x', s=100)
        axs[0].set_title(f"VAE Hall Correlation (Linear)\nProbe at {px},{py}")
        plt.colorbar(im1, ax=axs[0])
        
        im2 = axs[1].imshow(corr_p, cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
        axs[1].set_title("Cross-Correlation to Pedersen")
        plt.colorbar(im2, ax=axs[1])
        plt.show()

# --- 4. MAIN EXECUTION ---

if __name__ == "__main__":
    # 1. Load Data
    print("Loading Data using Helper...")
    H, P, W, t, o = helpers.load_data(PATHS, spar=CONFIG['SPARSITY_THRESHOLD'])
    
    subset = 3000
    H_trunc, P_trunc, W_trunc = H[:subset], P[:subset], W[:subset]
    
    # Calculate Stats for Normalization
    stats = {
        'H_mu': np.nanmean(H_trunc), 'H_sig': np.nanstd(H_trunc),
        'P_mu': np.nanmean(P_trunc), 'P_sig': np.nanstd(P_trunc)
    }
    
    # Normalize
    H_norm = (H_trunc - stats['H_mu']) / stats['H_sig']
    P_norm = (P_trunc - stats['P_mu']) / stats['P_sig']
    
    # Prepare Tensors
    X_np = np.stack([np.nan_to_num(H_norm, nan=0.0), 
                     np.nan_to_num(P_norm, nan=0.0), 
                     np.nan_to_num(W_trunc, nan=0.0)], axis=1)
    
    # Mask (1 Channel)
    M_np = (~(np.isnan(H_trunc) | np.isnan(P_trunc))).astype(np.float32)
    M_np = np.expand_dims(M_np, 1)
    
    dataset = TensorDataset(torch.from_numpy(X_np).float(), torch.from_numpy(M_np).float())
    loader = DataLoader(dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)
    
    # 2. Train VAE
    model = VAE2D(latent_dim=CONFIG['LATENT_DIM']).to(CONFIG['DEVICE'])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['LR'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)
    
    print("Starting Training...")
    losses = []
    model.train()
    
    for epoch in tqdm(range(CONFIG['EPOCHS'])):
        ep_loss = 0
        for x_batch, m_batch in loader:
            x_batch, m_batch = x_batch.to(CONFIG['DEVICE']), m_batch.to(CONFIG['DEVICE'])
            optimizer.zero_grad()
            
            recon, mu, logvar = model(x_batch, m_batch)
            loss, _, _ = loss_function(recon, x_batch, m_batch, mu, logvar, CONFIG['BETA'])
            
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
        
        avg_loss = ep_loss / len(loader)
        losses.append(avg_loss)
        scheduler.step(avg_loss)
    
    plt.plot(losses); plt.yscale('log'); plt.title('VAE Training Loss'); plt.show()
    
    # 3. Analyze Covariance
    analyzer = AuroralAnalyzer(model, stats)
    
    # Generate Linear Covariance
    analyzer.estimate_linear_covariance(n_samples=5000)
    
    # Plot 1: The Matrix
    analyzer.plot_covariance_matrix()
    
    # Plot 2: The Teleconnections
    analyzer.plot_teleconnections(probe_idx=(18*36 + 18))
    
    def plot_eigen_modes(cov_matrix, n_modes=5):
        """Visualize top Principal Components."""
        vals, vecs = scipy.linalg.eigh(cov_matrix)
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]
        vals_pct = vals / np.sum(vals) * 100
        
        fig, axs = plt.subplots(1, n_modes+1, figsize=(15, 4))
        fig.suptitle(f"Top {n_modes} Learned Modes of Variation (Eigen-Auroras)")
        
        for i in range(n_modes):
            mode_img = vecs[:36*36, i].reshape(36, 36)
            limit = np.max(np.abs(mode_img))
            axs[i].imshow(mode_img, cmap='RdBu_r', vmin=-limit, vmax=limit, origin='lower')
            axs[i].set_title(f"Mode {i+1}\n(Var: {vals_pct[i]:.2f}%)")
            axs[i].axis('off')
        axs[-1].plot(np.cumsum(vals_pct)[:100])
        plt.show()
        
    plot_eigen_modes(analyzer.cov_matrix)
    