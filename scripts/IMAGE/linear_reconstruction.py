#%% 1. CONFIGURATION & IMPORTS
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
import glob

import os
from icreader import ConductanceImage
os.chdir('/home/bing/Dropbox/work/code/repos/icAnalyzer/scripts/IMAGE')
import helpers 

# --- Hyperparameters ---
CONFIG = {
    'CHANNELS': 3,           # Hall, Pedersen, Weight
    'IMG_HEIGHT': 36,
    'IMG_WIDTH': 36,
    'LATENT_DIM': 36,
    'BATCH_SIZE': 8,
    'LR': 1e-4,
    'EPOCHS': 100,
    'SEED': 1,
    'SPARSITY_THRESHOLD': .33,
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# --- Paths ---
PATHS = {
    'base': '/home/bing/Dropbox/work',
    'data_in': '/home/bing/Dropbox/work/data/IMAGE_FUV/conductance',
    'segments': '/home/bing/Dropbox/work/code/repos/icAnalyzer/data/good_orbit_segments.csv'
}

np.random.seed(CONFIG['SEED'])
torch.manual_seed(CONFIG['SEED'])

#%% 2. MODEL ARCHITECTURE

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
    def __init__(self, latent_dim=128):
        super(VAE2D, self).__init__()
        self.latent_dim = latent_dim

        # --- ENCODER (3 Input Channels: H, P, W) ---
        self.enc_pconv1 = PartialConv2d(3, 16, kernel_size=3, stride=2, padding=1, return_mask=True)
        self.enc_bn1 = nn.BatchNorm2d(16)
        
        self.enc_pconv2 = PartialConv2d(16, 32, kernel_size=3, stride=2, padding=1, return_mask=True)
        self.enc_bn2 = nn.BatchNorm2d(32)
        
        self.enc_pconv3 = PartialConv2d(32, 64, kernel_size=3, stride=2, padding=1, return_mask=True)
        self.enc_bn3 = nn.BatchNorm2d(64)
        
        self.flatten_dim = 64 * 5 * 5 
        
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # --- DECODER ---
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        
        self.dec_conv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.dec_bn1 = nn.BatchNorm2d(32)
        
        self.dec_conv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn2 = nn.BatchNorm2d(16)
        
        # Outputs 3 Channels (H, P, W)
        self.dec_conv3 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

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
        x = x.view(x.size(0), 64, 5, 5) 
        x = F.leaky_relu(self.dec_bn1(self.dec_conv1(x)))
        x = F.leaky_relu(self.dec_bn2(self.dec_conv2(x)))
        x = self.dec_conv3(x) 
        return x

    def forward(self, x, mask):
        mu, logvar = self.encode(x, mask)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def loss_function(recon, x, mask, mu, logvar, beta=1):
    # x: [B, 3, H, W] -> (Hall, Pedersen, Weight)
    # recon: [B, 3, H, W] -> (Rec_Hall, Rec_Ped, Rec_Weight)

    # 1. Split Inputs
    target = x[:, :2, :, :]       # (B, 2, H, W)
    weights = x[:, 2:3, :, :]     # (B, 1, H, W)

    # 2. Split Outputs (We only calculate loss on Hall/Pedersen)
    recon_target = recon[:, :2, :, :] # (B, 2, H, W)

    # 3. Weights
    # Combined Weight: (Huber Weight) * (Binary Mask)
    final_weight = weights * mask

    # 4. MSE
    # (B, 2) - (B, 2) works now!
    recon_loss = torch.sum(final_weight * (recon_target - target)**2)
    
    # Normalization
    recon_loss = recon_loss / (final_weight.sum() + 1e-8) * (36*36*2)

    # KLD
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kld_loss

#%% 3. ANALYSIS TOOLS (Covariance, OI, Latent Opt)

def probe_latent_space(model, latent_dim, n_samples=5000):
    """Generate linear covariance matrix."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim).to(device)
        samples = model.decode(z)
        H = samples[:, 0, :, :].reshape(n_samples, -1).cpu().numpy()
        P = samples[:, 1, :, :].reshape(n_samples, -1).cpu().numpy()
        W = samples[:, 2, :, :].reshape(n_samples, -1).cpu().numpy()
        return H, P, W

def calculate_covariance_matrix(x):
    return np.cov(x, rowvar=False)

def calculate_joint_covariance_matrix(x1, x2):
    x = np.hstack([x1, x2])
    return calculate_covariance_matrix(x)

def perform_eigen_decomposition(x):
    """Visualize top Principal Components."""
    vals, vecs = scipy.linalg.eigh(x)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    vals_pct = vals / np.sum(vals) * 100
    return vecs, vals, vals_pct

def normalize(x):
    mu = np.nanmean(x)
    std = np.nanstd(x)
    x_norm = (x-mu) / std
    return x_norm, mu, std

def execute_VAE(H, P, W):
    # Create Masks and Inputs
    mask_np = (~(np.isnan(H) | np.isnan(P) | np.isnan(W))).astype(np.float32)
    
    # Stack H, P, W
    X_np = np.stack([np.nan_to_num(H, nan=0.0), 
                     np.nan_to_num(P, nan=0.0),
                     np.nan_to_num(W, nan=0.0)], axis=1)
    
    tensor_x = torch.from_numpy(X_np).float()
    tensor_mask = torch.from_numpy(mask_np).float().unsqueeze(1) # (N, 1, 36, 36)

    dataset = TensorDataset(tensor_x, tensor_mask)
    dataloader = DataLoader(dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)

    # --- 4.2 Initialize & Train Model ---
    model = VAE2D(latent_dim=CONFIG['LATENT_DIM']).to(CONFIG['DEVICE'])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['LR'])
    
    print(f"Training on {CONFIG['DEVICE']}...")
    model.train()
    for epoch in range(CONFIG['EPOCHS']):
        total_loss = 0
        for data, mask in dataloader:
            data, mask = data.to(CONFIG['DEVICE']), mask.to(CONFIG['DEVICE'])
            
            optimizer.zero_grad()
            recon, mu, logvar = model(data, mask)
            loss = loss_function(recon, data, mask, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{CONFIG['EPOCHS']} | Avg Loss: {total_loss/len(dataset):.4f}")
    
    return model

def reconstruct_from_sparse(model, sparse_data, sparse_mask, iterations=200, lr=0.1):
    """Latent Vector Optimization."""
    device = next(model.parameters()).device
    sparse_data, sparse_mask = sparse_data.to(device), sparse_mask.to(device)
    
    z = torch.randn(1, model.latent_dim, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=lr)
    model.eval()
    
    for i in range(iterations):
        optimizer.zero_grad()
        recon_full = model.decode(z) # (B, 3, H, W)
        
        # Slice to get only H/P
        recon_hp = recon_full[:, :2, :, :]
        
        loss = F.mse_loss(recon_hp * sparse_mask, sparse_data * sparse_mask, reduction='sum')
        loss = loss + 0.01 * torch.sum(z ** 2) # Regularization
        loss.backward()
        optimizer.step()
            
    return model.decode(z).detach().cpu().numpy()

def generate_vae_covariance(model, latent_dim, channel_idx=0, n_samples=5000):
    """Generate linear covariance matrix."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim).to(device)
        samples = model.decode(z)
        # Select channel (0=Hall, 1=Ped, 2=Weight)
        flat_samples = samples[:, channel_idx, :, :].reshape(n_samples, -1).cpu().numpy()
        cov_matrix = np.cov(flat_samples, rowvar=False)
        return cov_matrix, flat_samples

def generate_log_statistics(model, latent_dim, P_stats, channel_idx=1, n_samples=5000, epsilon=1.0):
    """Generate Log-Space statistics for Log-OI."""
    P_mu, P_sig = P_stats
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim).to(device)
        # Select channel
        samples_norm = model.decode(z)[:, channel_idx, :, :] 
        
        # De-Normalize to Physical Units
        samples_phys = (samples_norm.cpu().numpy() * P_sig) + P_mu
        samples_phys = np.maximum(samples_phys, 0)
        
        # Log Transform
        samples_log = np.log(samples_phys + epsilon)
        samples_flat = samples_log.reshape(n_samples, -1)
        
        mean_log = np.mean(samples_flat, axis=0)
        cov_log = np.cov(samples_flat, rowvar=False)
        return mean_log, cov_log

def log_OI(mean, cov, y, y_std, mask, n_modes=50):
    vals, vecs = scipy.linalg.eigh(cov)
    
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx][:n_modes], vecs[:, idx][:, :n_modes]
    P = (vecs * vals) @ vecs.T
    
    # Solve OI
    P_Ht = P[:, mask]
    H_P_Ht = P[np.ix_(mask, mask)]
    
    R = np.diag(y_std**2)
    
    S = H_P_Ht + R + np.eye(len(y)) * 1e-6
    K = scipy.linalg.solve(S, P_Ht.T).T
    
    x_analysis_log = mean + K @ (y - mean[mask])
    
    # Convert back to Physical
    x_phys = np.maximum(np.exp(x_analysis_log) - 1, 0.0)
    
    return x_phys

def linear_OI(mean, cov, y, y_std, mask, n_modes=50):
    """
    Linear Space Optimal Interpolation.
    y: physical observations
    y_std: physical observation error (e.g., 0.5)
    """
    # 1. Reduced Rank Covariance
    vals, vecs = scipy.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx][:n_modes], vecs[:, idx][:, :n_modes]
    P = (vecs * vals) @ vecs.T
    
    # 2. Setup Matrices
    P_Ht = P[:, mask]
    H_P_Ht = P[np.ix_(mask, mask)]
    
    R = np.diag(y_std**2)
    
    # 3. Calculate Kalman Gain
    S = H_P_Ht + R + np.eye(len(y)) * 1e-6
    K = scipy.linalg.solve(S, P_Ht.T).T
    
    # 4. State Update
    x_analysis = mean + K @ (y - mean[mask])
    
    # 5. Enforce Physics (Clip negatives)
    #x_phys = np.maximum(x_analysis, 0.0)
    
    return x_analysis

def illustrate_PCs(x, pct):
    vmax = np.max(abs(x[:, :10]))
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(x[:, i].reshape(CONFIG['IMG_HEIGHT'], CONFIG['IMG_WIDTH']),
                   vmin=-vmax, vmax=vmax, cmap='bwr', origin='lower')
        ax.set_title(f'PC {i+1} : {np.round(pct[i], 1)}%')
        ax.set_axis_off()
    return fig, axs

#%% 4. MAIN EXECUTION BLOCK

if __name__ == "__main__":
    
    print('Loading data')
    H, P, W, t, o = helpers.load_data(PATHS, spar=CONFIG['SPARSITY_THRESHOLD'])

    



    print('Preparing data')
    H_norm, H_mu, H_std = normalize(H)
    P_norm, P_mu, P_std = normalize(P)
    
    
    
    subset_size = 3000
    H_trunc = H_norm[:subset_size]
    P_trunc = P_norm[:subset_size]
    W_trunc = W[:subset_size]
    
    print('Running VAE')
    model = execute_VAE(H_trunc, P_trunc, W_trunc)
    
    print('Probing latent space')
    H_samp, P_samp, W_samp = probe_latent_space(model, CONFIG['LATENT_DIM'], n_samples=100000)
    
    print('Calculating covariance matrices')
    cov_H = calculate_covariance_matrix(H_samp)
    cov_P = calculate_covariance_matrix(P_samp)
    cov   = calculate_joint_covariance_matrix(H_samp, P_samp)

    print('Performing eigen decomposition')
    H_vecs, H_vals, H_vals_pct = perform_eigen_decomposition(cov_H)
    P_vecs, P_vals, P_vals_pct = perform_eigen_decomposition(cov_P)
    vecs, vals, vals_pct       = perform_eigen_decomposition(cov)

    print('Comparing explained variance')
    plt.figure(figsize=(15, 9))
    plt.plot(np.cumsum(P_vals_pct), label='Pedersen')
    plt.plot(np.cumsum(H_vals_pct), label='Hall')
    plt.plot(np.cumsum(vals_pct), label='Joint')
    plt.xlabel('PC')
    plt.ylabel('Variance explained [%]')
    plt.title('Individual and joint covariance comparison')
    plt.legend()
    plt.xlim(-1, 250)
    plt.ylim(30, 101)
    plt.grid()

    fig, axs = illustrate_PCs(H_vecs, H_vals_pct)
    plt.suptitle('PCA of Hall covariance')
    fig, axs = illustrate_PCs(P_vecs, P_vals_pct)
    plt.suptitle('PCA of Pedersen covariance')
    fig, axs = illustrate_PCs(vecs[:36*36], vals_pct)
    plt.suptitle('PCA of joint covariance (Hall)')
    fig, axs = illustrate_PCs(vecs[36*36:], vals_pct)
    plt.suptitle('PCA of joint covariance (Pedersen)')
    
    ######## LOG
    eps = 1
    H_samp_log = np.log(np.maximum((H_samp * H_std) + H_mu, 0) + eps)
    P_samp_log = np.log(np.maximum((P_samp * P_std) + P_mu, 0) + eps)
    
    print('Calculating covariance matrices')
    H_log_mu = np.mean(H_samp_log, axis=0)
    P_log_mu = np.mean(P_samp_log, axis=0)
    
    cov_H = calculate_covariance_matrix(H_samp_log)
    cov_P = calculate_covariance_matrix(P_samp_log)
    cov   = calculate_joint_covariance_matrix(H_samp_log, P_samp_log)

    print('Performing eigen decomposition')
    H_vecs, H_vals, H_vals_pct = perform_eigen_decomposition(cov_H)
    P_vecs, P_vals, P_vals_pct = perform_eigen_decomposition(cov_P)
    vecs, vals, vals_pct       = perform_eigen_decomposition(cov)

    print('Comparing explained variance')
    plt.figure(figsize=(15, 9))
    plt.plot(np.cumsum(P_vals_pct), label='Pedersen')
    plt.plot(np.cumsum(H_vals_pct), label='Hall')
    plt.plot(np.cumsum(vals_pct), label='Joint')
    plt.xlabel('PC')
    plt.ylabel('Variance explained [%]')
    plt.title('Individual and joint covariance comparison')
    plt.legend()
    plt.xlim(-1, 250)
    plt.ylim(30, 101)
    plt.grid()

    fig, axs = illustrate_PCs(H_vecs, H_vals_pct)
    plt.suptitle('PCA of Hall covariance')
    fig, axs = illustrate_PCs(P_vecs, P_vals_pct)
    plt.suptitle('PCA of Pedersen covariance')
    fig, axs = illustrate_PCs(vecs[:36*36], vals_pct)
    plt.suptitle('PCA of joint covariance (Hall)')
    fig, axs = illustrate_PCs(vecs[36*36:], vals_pct)
    plt.suptitle('PCA of joint covariance (Pedersen)')
    
    idx = 3600
    H_test, P_test = H[idx].flatten(), P[idx].flatten()
    drop_idx = np.unique(np.random.randint(0, 36*36, 3000)) # Heavy sparsification
    H_test[drop_idx] = np.nan
    P_test[drop_idx] = np.nan
    
    mask = ~(np.isnan(H_test) | np.isnan(P_test))
    H_sub, P_sub = H_test[mask], P_test[mask]
    
    rows = np.arange(36*36)[mask] // 36
    cols = np.arange(36*36)[mask] % 36
    
    vmin = 0
    vmax = np.nanmax(P_test)
    plt.figure()
    plt.imshow(P_test.reshape(36, 36), cmap='plasma', origin='lower', vmin=vmin, vmax=vmax)
    plt.scatter(cols, rows, 1, facecolor='k')
    
    for i in np.arange(0, 2, .1):
        H_sub_log_std = i / (H_sub + eps)
        P_sub_log_std = i / (P_sub + eps)
    
        H_sub_log = np.log(H_sub + eps)
        P_sub_log = np.log(P_sub + eps)
    
        P_fit = log_OI(P_log_mu, cov_P, P_sub_log, P_sub_log_std, mask, n_modes=50)

        plt.figure()
        plt.imshow(P_fit.reshape(36, 36), cmap='plasma', origin='lower', vmin=vmin, vmax=vmax)
        plt.scatter(cols, rows, 1, facecolor='k')
    
    
    
    
    
    
    
    
    
    
    
    

