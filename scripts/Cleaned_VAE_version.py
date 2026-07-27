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

# Try importing custom library, handle failure gracefully
try:
    from icreader import ConductanceImage
except ImportError:
    print("Warning: 'icreader' not found. Data loading block may fail.")

# --- Hyperparameters ---
CONFIG = {
    'CHANNELS': 2,           # Hall, Pedersen
    'IMG_HEIGHT': 36,
    'IMG_WIDTH': 36,
    'LATENT_DIM': 36,
    'BATCH_SIZE': 8,
    'LR': 1e-4,
    'EPOCHS': 100,
    'SEED': 1,
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
        # Handle custom args
        self.multi_channel = kwargs.pop('multi_channel', False)
        self.return_mask = kwargs.pop('return_mask', False)
        
        super(PartialConv2d, self).__init__(*args, **kwargs)

        # Fixed kernel of 1s for mask updating
        self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
        self.slide_winsize = self.kernel_size[0] * self.kernel_size[1]
        self.last_size = (0, 0, 0, 0)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        # FIX: Ensure mask is always 1-channel for broadcasting
        if mask_in is not None and mask_in.shape[1] > 1:
            mask_in = mask_in[:, 0:1, :, :]
            
        with torch.no_grad():
            if self.weight_maskUpdater.type() != input.type():
                self.weight_maskUpdater = self.weight_maskUpdater.to(input)

            if mask_in is None:
                mask_in = torch.ones(input.size(0), 1, input.size(2), input.size(3)).to(input)

            # Update Mask: Count valid pixels in each window
            update_mask = F.conv2d(mask_in, self.weight_maskUpdater, bias=None, 
                                   stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

            # Calculate scaling factor
            self.mask_ratio = self.slide_winsize / (update_mask + 1e-8)
            self.update_mask = torch.clamp(update_mask, 0, 1)
            self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        # Standard Convolution
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

        # --- ENCODER (Partial Conv) ---
        self.enc_pconv1 = PartialConv2d(2, 16, kernel_size=3, stride=2, padding=1, return_mask=True)
        self.enc_bn1 = nn.BatchNorm2d(16)
        
        self.enc_pconv2 = PartialConv2d(16, 32, kernel_size=3, stride=2, padding=1, return_mask=True)
        self.enc_bn2 = nn.BatchNorm2d(32)
        
        self.enc_pconv3 = PartialConv2d(32, 64, kernel_size=3, stride=2, padding=1, return_mask=True)
        self.enc_bn3 = nn.BatchNorm2d(64)
        
        self.flatten_dim = 64 * 5 * 5 
        
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # --- DECODER (Standard Conv) ---
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        
        self.dec_conv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.dec_bn1 = nn.BatchNorm2d(32)
        
        self.dec_conv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn2 = nn.BatchNorm2d(16)
        
        self.dec_conv3 = nn.ConvTranspose2d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=1)

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
    # Masked MSE
    recon_loss = F.mse_loss(recon * mask, x * mask, reduction='sum')
    
    # Normalize by valid pixels
    valid_count = mask.sum() + 1e-8
    recon_loss = recon_loss / valid_count * (x.shape[2] * x.shape[3])
    
    # KLD
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kld_loss

#%% 3. ANALYSIS TOOLS (Covariance, OI, Latent Opt)

def reconstruct_from_sparse(model, sparse_data, sparse_mask, iterations=200, lr=0.1):
    """Latent Vector Optimization."""
    device = next(model.parameters()).device
    sparse_data, sparse_mask = sparse_data.to(device), sparse_mask.to(device)
    
    z = torch.randn(1, model.latent_dim, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=lr)
    model.eval()
    
    for i in range(iterations):
        optimizer.zero_grad()
        recon = model.decode(z)
        loss = F.mse_loss(recon * sparse_mask, sparse_data * sparse_mask, reduction='sum')
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

def log_space_oi_with_uncertainty(log_mean, log_cov, sparse_data_norm, sparse_mask, P_stats, 
                                  error_std_phys=0.5, epsilon=1.0, n_modes=50):
    """Log-Space Optimal Interpolation with Uncertainty."""
    P_mu, P_sig = P_stats
    h, w = 36, 36
    
    # Prepare Data
    y_norm = sparse_data_norm[0, 1].cpu().numpy().flatten()
    mask_flat = sparse_mask[0, 0].cpu().numpy().flatten()
    valid_idx = np.where(mask_flat > 0)[0]
    
    y_phys = (y_norm[valid_idx] * P_sig) + P_mu
    y_phys = np.maximum(y_phys, 0)
    y_log = np.log(y_phys + epsilon)
    
    # Reduced Rank Covariance
    vals, vecs = scipy.linalg.eigh(log_cov)
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx][:n_modes], vecs[:, idx][:, :n_modes]
    P_k = (vecs * vals) @ vecs.T
    
    prior_var_log = np.diag(P_k)
    
    # Solve OI
    x_b = log_mean
    P_Ht = P_k[:, valid_idx]
    H_P_Ht = P_k[np.ix_(valid_idx, valid_idx)]
    
    sigma_log_obs = error_std_phys / (y_phys + epsilon)
    R = np.diag(sigma_log_obs**2)
    
    S = H_P_Ht + R + np.eye(len(valid_idx)) * 1e-6
    K = scipy.linalg.solve(S, P_Ht.T).T
    
    x_analysis_log = x_b + K @ (y_log - x_b[valid_idx])
    
    # Uncertainty
    explained_var = np.sum(K * P_Ht, axis=1)
    analysis_var_log = np.maximum(prior_var_log - explained_var, 0)
    
    # Convert back to Physical
    x_phys = np.maximum(np.exp(x_analysis_log) - epsilon, 0)
    sigma_factor = np.sqrt(np.exp(analysis_var_log) - 1)
    sigma_phys = (x_phys + epsilon) * sigma_factor
    
    return x_phys.reshape(h, w), sigma_phys.reshape(h, w)

def plot_eigen_modes(cov_matrix, n_modes=5):
    """Visualize top Principal Components."""
    vals, vecs = scipy.linalg.eigh(cov_matrix)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    vals_pct = vals / np.sum(vals) * 100
    
    fig, axs = plt.subplots(1, n_modes, figsize=(15, 4))
    fig.suptitle(f"Top {n_modes} Learned Modes of Variation (Eigen-Auroras)")
    
    for i in range(n_modes):
        mode_img = vecs[:, i].reshape(36, 36)
        limit = np.max(np.abs(mode_img))
        axs[i].imshow(mode_img, cmap='RdBu_r', vmin=-limit, vmax=limit, origin='lower')
        axs[i].set_title(f"Mode {i+1}\n(Var: {vals_pct[i]:.2f}%)")
        axs[i].axis('off')
    plt.show()

#%% 4. MAIN EXECUTION BLOCK

if __name__ == "__main__":
    
    # --- 4.1 Load and Process Data ---
    # This block assumes files exist. 
    gos = pd.read_csv(PATHS['segments'], header=None)
    gos.columns = ['orbit', 'shit_start', 'shit_stop', 'good_start', 'good_stop']
    gos = gos.loc[gos['orbit'] < 503, :]

    t, H, P, o = [], [], [], []
    for orbit in tqdm(gos['orbit'], total=gos.shape[0]):
        try:
            cI = ConductanceImage(pjoin(PATHS['data_in'], f'or_{str(orbit).zfill(4)}.nc'))
            start = gos.loc[gos['orbit']==orbit].iloc[0]['good_start']
            stop = gos.loc[gos['orbit']==orbit].iloc[0]['good_stop']
            
            if stop == 0 or np.isnan(start): continue
            if np.isnan(stop): stop = cI.H.shape[0]-1
            
            start, stop = int(start), int(stop+1)
            H.append(cI.H[start:stop])
            P.append(cI.P[start:stop])
            t.append(cI.time[start:stop])
            o.append(np.ones(cI.time[start:stop].size)*orbit)
        except Exception as e:
            continue

    H = np.concatenate(H, axis=0)
    P = np.concatenate(P, axis=0)
    
    # Filtering empty frames
    h, w = CONFIG['IMG_HEIGHT'], CONFIG['IMG_WIDTH']
    f = np.sum((np.isnan(H) | np.isnan(P)), axis=(1,2)) / (h*w)
    f = f < .33
    H, P = H[f], P[f]

    # Normalization Stats (Keep these for reconstruction!)
    H_mu, H_sig = np.nanmean(H), np.nanstd(H)
    P_mu, P_sig = np.nanmean(P), np.nanstd(P)
    
    # Normalize & Truncate
    H_norm = (H - H_mu) / H_sig
    P_norm = (P - P_mu) / P_sig
    
    # Prepare Tensor Data
    subset_size = 3000
    H_trunc = H_norm[:subset_size]
    P_trunc = P_norm[:subset_size]
    
    # Create Masks and Inputs
    mask_np = (~(np.isnan(H_trunc) | np.isnan(P_trunc))).astype(np.float32)
    X_np = np.stack([np.nan_to_num(H_trunc, nan=0.0), np.nan_to_num(P_trunc, nan=0.0)], axis=1)
    
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

    # --- 4.3 Analysis: Covariance & PCA ---
    print("Generating covariance...")
    cov_matrix, _ = generate_vae_covariance(model, CONFIG['LATENT_DIM'], channel_idx=1)
    plot_eigen_modes(cov_matrix)
    
    # --- 4.4 Analysis: Log-Space Reconstruction ---
    print("Running Log-Space OI...")
    
    # Create a sparse test case
    id_ = 3600
    H_flat, P_flat = H_norm[id_].flatten(), P_norm[id_].flatten()
    drop_idx = np.unique(np.random.randint(0, 36*36, 3000)) # Heavy sparsification
    H_flat[drop_idx] = np.nan
    P_flat[drop_idx] = np.nan
    
    mask_sparse = (~(np.isnan(H_flat) | np.isnan(P_flat))).astype(np.float32).reshape(36, 36)
    H_sparse = np.nan_to_num(H_flat.reshape(36, 36), nan=0.0)
    P_sparse = np.nan_to_num(P_flat.reshape(36, 36), nan=0.0)
    
    sparse_data_tensor = torch.from_numpy(np.stack([H_sparse, P_sparse])).float().unsqueeze(0)
    sparse_mask_tensor = torch.from_numpy(mask_sparse).float().unsqueeze(0).unsqueeze(0)

    # Calculate Log Stats
    log_mean, log_cov = generate_log_statistics(model, CONFIG['LATENT_DIM'], (P_mu, P_sig), channel_idx=1)
    
    # Run Reconstruction
    rec_phys, unc_phys = log_space_oi_with_uncertainty(
        log_mean, log_cov, 
        sparse_data_tensor, sparse_mask_tensor, 
        P_stats=(P_mu, P_sig),
        error_std_phys=0.5, 
        epsilon=1.0, 
        n_modes=50
    )

    # Plot Final Results
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    
    # Ground Truth
    gt_phys = (P_norm[id_] * P_sig) + P_mu
    im = axs[0].imshow(gt_phys, cmap='plasma', origin='lower', vmin=0)
    axs[0].set_title("Ground Truth")
    plt.colorbar(im, ax=axs[0])
    
    # Input (Sparse)
    in_phys = (sparse_data_tensor[0,1].numpy() * P_sig) + P_mu
    in_phys[mask_sparse==0] = np.nan
    axs[1].imshow(in_phys, cmap='plasma', origin='lower', vmin=0, vmax=np.nanmax(gt_phys))
    axs[1].set_title("Sparse Input")
    
    # Reconstruction
    im = axs[2].imshow(rec_phys, cmap='plasma', origin='lower', vmin=0, vmax=np.nanmax(gt_phys))
    axs[2].set_title("Log-OI Reconstruction")
    plt.colorbar(im, ax=axs[2])
    
    # Uncertainty
    im = axs[3].imshow(unc_phys, cmap='viridis', origin='lower')
    axs[3].set_title("Uncertainty (Sigma)")
    plt.colorbar(im, ax=axs[3])
    
    plt.tight_layout()
    plt.show()
    
    print(f"Reconstruction Min Value: {np.min(rec_phys):.4f} (Should be >= 0)")