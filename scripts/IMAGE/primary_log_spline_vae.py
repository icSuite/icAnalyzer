"""Primary working example for the log-space spline VAE.

This script is derived from ``doing_it_manually_log_spline.py``. It fixes
cross-channel covariance normalization and performs latent assimilation in
the same normalized log10 space used to train the model.

Deferred work:
* restore independent-versus-joint observation experiments;
* add adaptive rank selection, posterior uncertainty, orbit-separated
  validation, quantitative metrics, and reproducible run artifacts.

The iterative-PCA baseline is intentionally not part of this lineage.
"""

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
import copy
from scipy.interpolate import BSpline

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
    'EPOCHS': 1000,
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

class SplineVAE2D(nn.Module):
    def __init__(self, latent_dim=128):
        super(SplineVAE2D, self).__init__()
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

        # --- DECODER (Architectural Fix for Checkerboard Artifacts) ---
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        
        # Block 1: 5x5 -> 9x9 (Nearest neighbor resizing, then smooth convolution)
        self.up1 = nn.Upsample(size=(9, 9), mode='nearest')
        self.dec_conv1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(32)
        
        # Block 2: 9x9 -> 18x18
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec_conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(16)
        
        # --- THE FULLY CONVOLUTIONAL SPLINE LAYER ---
        # Dynamically resizes the 18x18 feature map to exactly ncp x ncp
        self.spline_resize = nn.AdaptiveAvgPool2d((ncp, ncp)) 
        
        # Outputs exactly 2 channels (Hall, Pedersen) of size ncp x ncp
        self.spline_conv = nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1)
        self.softplus = nn.Softplus()

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

    def decode(self, z, device):
        # 1. Standard Spatial Decoding
        x = self.fc_decode(z)
        x = x.view(x.size(0), 64, 5, 5) 
        
        x = self.up1(x)
        x = F.leaky_relu(self.dec_bn1(self.dec_conv1(x)))
        
        x = self.up2(x)
        x = F.leaky_relu(self.dec_bn2(self.dec_conv2(x))) # Shape: [B, 16, 18, 18]
        
        # 2. Fully Convolutional Spline Output
        x = self.spline_resize(x)               # Shape: [B, 16, ncp, ncp]
        c_grid = self.spline_conv(x)            # Shape: [B, 2, ncp, ncp]
        #c_grid = self.softplus(c_grid)          # Physical constraint!
        
        # Flatten the 2D spatial grids into 1D lists for the G matrix multiplication
        B = c_grid.size(0)
        c_H = c_grid[:, 0, :, :].reshape(B, -1) # [B, ncp^2]
        c_P = c_grid[:, 1, :, :].reshape(B, -1) # [B, ncp^2]

        # 3. Pull in Global Variables
        G_t = torch.tensor(G, dtype=torch.float32, device=device)
        H_mu_t = torch.tensor(H_mu, dtype=torch.float32, device=device)
        H_std_t = torch.tensor(H_std, dtype=torch.float32, device=device)
        P_mu_t = torch.tensor(P_mu, dtype=torch.float32, device=device)
        P_std_t = torch.tensor(P_std, dtype=torch.float32, device=device)

        # 4. Project to Image
        H_phys_flat = torch.matmul(c_H, G_t.T) 
        P_phys_flat = torch.matmul(c_P, G_t.T) 

        H_phys = H_phys_flat.view(-1, 1, 36, 36)
        P_phys = P_phys_flat.view(-1, 1, 36, 36)

        # 5. Normalize and Package
        H_norm = (H_phys - H_mu_t) / H_std_t
        P_norm = (P_phys - P_mu_t) / P_std_t

        dummy_W = torch.zeros_like(H_norm) 
        recon = torch.cat([H_norm, P_norm, dummy_W], dim=1) 
        
        # [NEW] Return the coefficients alongside the reconstructed image!
        return recon, c_H, c_P

    def forward(self, x, mask):
        mu, logvar = self.encode(x, mask)
        z = self.reparameterize(mu, logvar)
        
        # Unpack all three outputs from decode...
        recon, c_H, c_P = self.decode(z, x.device)
        
        # ...but ONLY return what the training loop expects!
        # This prevents your execute_VAE loop from crashing.
        return recon, mu, logvar

def loss_function(recon, x, mask, mu, logvar, beta=1):
    
    # 1. Standard MSE on normalized data
    # Slicing [:2] guarantees we only look at Hall and Pedersen, ignoring the dummy Weight channel
    recon_target = recon[:, :2, :, :] 
    target = x[:, :2, :, :]
    weights = x[:, 2:3, :, :] * mask
    
    mse_loss = torch.sum(weights * (recon_target - target)**2) / (weights.sum() + 1e-8) * (36*36*2)

    # 2. KLD
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Positivity is enforced after transforming reconstructed log10 fields
    # back to physical conductance.
    return mse_loss + (beta * kld_loss)

#%% Conductance Image class

class Spline:
    """
    Attributes
    ----------
    grid : CSgrid
        Cubbed sphere grid on which the data is projected.
    k : int
        Spatial spline degree.
    nk : int
        Amount of knots in spatial spline.
        
    """    
    def __init__(self, grid, nk, k):
        self.grid = grid
        self.nk = nk
        self.k = k
        self._G = None        
        self.init_splines()

    @property
    def x(self):
        return self.grid.xi.flatten()
    
    @property
    def y(self):
        return self.grid.eta.flatten()

    @property
    def ncp(self):
        return self.nk - self.k - 1

    @property
    def knots(self):
        return np.r_[[self.x.min()]*self.k, np.linspace(self.x.min(), self.x.max(), self.nk-2*self.k), [self.x.max()]*self.k]
    
    @property
    def sply(self):
        return self.splx

    @property
    def G(self):
        if self._G is None:
            self._G = self.generate_G_2d()
        return self._G

    def init_splines(self):
        self.splx = BSpline(self.knots, np.eye(self.ncp), self.k)

    def generate_G_2d(self):
        """
        Optimized generation of the Design Matrix (G).
        Speedup: ~100x-1000x compared to looping.
        """
        # 1. Get Basis matrices for ALL points at once
        # Scipy BSpline evaluates the whole array efficiently in C
        # Shape: (N_points, ncp)
        Bx = self.splx(self.x.flatten()) 
        By = self.sply(self.y.flatten())

        # 2. Compute Row-wise Kronecker product via Broadcasting
        # We want: G[i] = Bx[i] (outer_product) By[i]
        
        # Reshape to allow broadcasting: 
        # (N, ncp, 1) * (N, 1, ncp) -> (N, ncp, ncp)
        G_3d = Bx[:, :, np.newaxis] * By[:, np.newaxis, :]
        
        # 3. Flatten the last two dimensions to get the 2D design matrix
        # Shape: (N_points, ncp * ncp)
        G = G_3d.reshape(self.x.size, self.ncp**2)

        return G

#%% 3. ANALYSIS TOOLS (Covariance, OI, Latent Opt)

def probe_latent_space_old(model, latent_dim, n_samples=5000):
    """Generate linear covariance matrix."""
    # 1. Draw a random sample from the standard normal distribution
    z_sample = torch.randn(n_samples, latent_dim).to(CONFIG['DEVICE'])

    # 2. Call decode directly to get your spline coefficients
    model.eval()
    with torch.no_grad():
        recon_image, hall_splines, pedersen_splines = model.decode(z_sample, CONFIG['DEVICE'])

    # Convert to numpy for your science
    c_H = hall_splines.cpu().numpy() # Shape: (1, 676)
    c_P = pedersen_splines.cpu().numpy() # Shape: (1, 676)

    print("Extracted Hall Control Points:", c_H.shape)
    return c_H, c_P

def probe_latent_space(model, latent_dim, n_samples=5000, chunk_size=500):
    """Generate linear covariance matrix safely in chunks."""
    model.eval()
    
    all_cH = []
    all_cP = []
    
    with torch.no_grad():
        # Loop through in safe chunks so the GPU never chokes
        for _ in range(0, n_samples, chunk_size):
            # Draw a smaller chunk
            z_chunk = torch.randn(chunk_size, latent_dim).to(CONFIG['DEVICE'])
            
            # Decode just the chunk
            _, c_H, c_P = model.decode(z_chunk, CONFIG['DEVICE'])
            
            # Instantly move to CPU and append to list
            all_cH.append(c_H.cpu().numpy())
            all_cP.append(c_P.cpu().numpy())

    # Glue the numpy arrays together
    final_cH = np.concatenate(all_cH, axis=0)
    final_cP = np.concatenate(all_cP, axis=0)

    print("Extracted Hall Control Points:", final_cH.shape)
    
    # Optional: Force PyTorch to release the VRAM back to the OS
    torch.cuda.empty_cache() 
    
    return final_cH, final_cP

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

def execute_VAE(H, P, W, beta=1):
    # Create Masks and Inputs
    mask_np = (~(np.isnan(H) | np.isnan(P) | np.isnan(W))).astype(np.float32)
    
    # Stack H, P, W
    X_np = np.stack([np.nan_to_num(H, nan=0.0), 
                     np.nan_to_num(P, nan=0.0),
                     np.nan_to_num(W, nan=0.0)], axis=1)
    
    tensor_x = torch.from_numpy(X_np).float()
    tensor_mask = torch.from_numpy(mask_np).float().unsqueeze(1) # (N, 1, 36, 36)
    #tensor_x = torch.from_numpy(X_np).float().to(CONFIG['DEVICE'])
    #tensor_mask = torch.from_numpy(mask_np).float().unsqueeze(1).to(CONFIG['DEVICE'])

    dataset = TensorDataset(tensor_x, tensor_mask)
    #dataloader = DataLoader(dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)
    dataloader = DataLoader(dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True,
                            num_workers=4,        # Uses 4 CPU cores to prep the next batch
                            pin_memory=True,      # Pre-allocates memory for lightning-fast PCIe transfer
                            persistent_workers=True)

    # --- 4.2 Initialize & Train Model ---
    model = SplineVAE2D(latent_dim=CONFIG['LATENT_DIM']).to(CONFIG['DEVICE'])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['LR'])
    
    # [NEW] Initialize the scheduler
    # factor=0.5: cuts LR in half when plateauing
    # patience=10: waits 10 epochs of no improvement before dropping LR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    print(f"Training on {CONFIG['DEVICE']}...")
    losses = []
    lrs = []
    model.train()
    for epoch in range(CONFIG['EPOCHS']):
        total_loss = 0
        for data, mask in dataloader:
            data, mask = data.to(CONFIG['DEVICE']), mask.to(CONFIG['DEVICE'])
            
            optimizer.zero_grad()
            recon, mu, logvar = model(data, mask)
            loss = loss_function(recon, data, mask, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Calculate average loss for this epoch
        avg_loss = total_loss / len(dataset)
        losses.append(avg_loss) # [CHANGED] Storing avg_loss is better for plotting
            
        # [NEW] Step the scheduler based on the epoch's average loss
        scheduler.step(avg_loss)
        
        # [NEW] Optional: Grab the current LR to print it, so you can see when it drops
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        print(f"Epoch {epoch+1}/{CONFIG['EPOCHS']} | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
    
    return model, losses, lrs

def log10_OI(mean, cov, y, y_std, mask, n_modes=50, epsilon=1.0):
    """Optimal interpolation for a state represented as log10(x + epsilon)."""
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
    
    x_analysis_log10 = mean + K @ (y - mean[mask])
    
    # Convert back to Physical
    x_phys = np.maximum(10**x_analysis_log10 - epsilon, 0.0)
    
    return x_phys

def illustrate_PCs(x, pct):
    n = int(np.sqrt(x.shape[0]))
    vmax = np.max(abs(x[:, :10]))
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(x[:, i].reshape(n, n),
                   vmin=-vmax, vmax=vmax, cmap='bwr', origin='lower')
        ax.set_title(f'PC {i+1} : {np.round(pct[i], 1)}%')
        ax.set_axis_off()
    return fig, axs

def covariance_to_correlation(cov_matrix, row_variances=None, column_variances=None):
    """Normalize a covariance or cross-covariance block.

    Cross-covariance blocks do not contain marginal variances on their
    diagonals. Callers must therefore supply the variances from the matching
    row and column marginal covariance blocks.
    """
    cov_matrix = np.asarray(cov_matrix)
    if cov_matrix.ndim != 2:
        raise ValueError("cov_matrix must be two-dimensional")

    if row_variances is None or column_variances is None:
        if cov_matrix.shape[0] != cov_matrix.shape[1]:
            raise ValueError(
                "row_variances and column_variances are required for "
                "non-square cross-covariance blocks"
            )
        marginal_variances = np.diag(cov_matrix)
        if row_variances is None:
            row_variances = marginal_variances
        if column_variances is None:
            column_variances = marginal_variances

    row_variances = np.asarray(row_variances)
    column_variances = np.asarray(column_variances)
    if row_variances.shape != (cov_matrix.shape[0],):
        raise ValueError("row_variances has the wrong shape")
    if column_variances.shape != (cov_matrix.shape[1],):
        raise ValueError("column_variances has the wrong shape")

    scale = np.sqrt(
        np.maximum(row_variances, 0.0)[:, None]
        * np.maximum(column_variances, 0.0)[None, :]
    )
    correlation = np.zeros_like(cov_matrix, dtype=float)
    np.divide(cov_matrix, scale, out=correlation, where=scale > 0)
    return np.clip(correlation, -1.0, 1.0)


def get_teleconnection(
    cov_matrix,
    base_pixel_idx,
    row_variances=None,
    column_variances=None,
):
    """Return the correlation map for one row of a covariance block."""
    correlation_matrix = covariance_to_correlation(
        cov_matrix,
        row_variances=row_variances,
        column_variances=column_variances,
    )
    teleconnections = correlation_matrix[base_pixel_idx, :]

    n = int(np.sqrt(cov_matrix.shape[1]))
    if n * n != cov_matrix.shape[1]:
        raise ValueError("The covariance columns cannot be reshaped to a square map")

    return teleconnections.reshape(n, n)


def _normalized_log10_observations(values, error_std_phys, mean, std, epsilon):
    """Transform physical observations and errors to normalized log10 units."""
    values = np.asarray(values, dtype=float)
    if np.any(values + epsilon <= 0):
        raise ValueError("Physical observations must satisfy value + epsilon > 0")
    if std <= 0:
        raise ValueError("Normalization standard deviation must be positive")

    error_std_phys = np.broadcast_to(
        np.asarray(error_std_phys, dtype=float), values.shape
    )
    if np.any(error_std_phys <= 0):
        raise ValueError("Physical observation errors must be positive")

    values_log10 = np.log10(values + epsilon)
    values_normalized = (values_log10 - mean) / std

    # First-order propagation through log10(x + epsilon), followed by the
    # training normalization.
    errors_log10 = error_std_phys / ((values + epsilon) * np.log(10.0))
    errors_normalized = errors_log10 / std
    errors_normalized = np.maximum(errors_normalized, 1e-6)
    return values_normalized, errors_normalized


def latent_space_assimilation(
    model,
    H_observed,
    P_observed,
    mask,
    spline_matrix,
    H_stats,
    P_stats,
    *,
    observation_error_phys=0.3,
    epsilon=1.0,
    n_steps=100,
    lr=0.1,
):
    """Assimilate sparse physical observations by optimizing the latent state.

    The observation term is evaluated in normalized log10 space, matching the
    model's training targets. Returned Hall and Pedersen fields are converted
    back to physical conductance.
    """
    model.eval()
    device = next(model.parameters()).device
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (36 * 36,):
        raise ValueError("mask must have shape (1296,)")

    H_mu_local, H_std_local = H_stats
    P_mu_local, P_std_local = P_stats
    H_target, H_error = _normalized_log10_observations(
        H_observed,
        observation_error_phys,
        H_mu_local,
        H_std_local,
        epsilon,
    )
    P_target, P_error = _normalized_log10_observations(
        P_observed,
        observation_error_phys,
        P_mu_local,
        P_std_local,
        epsilon,
    )
    if H_target.size != mask.sum() or P_target.size != mask.sum():
        raise ValueError("Observed arrays must contain one value per valid mask pixel")

    target = torch.as_tensor(
        np.hstack((H_target, P_target)),
        dtype=torch.float32,
        device=device,
    )
    observation_error = torch.as_tensor(
        np.hstack((H_error, P_error)),
        dtype=torch.float32,
        device=device,
    )
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)

    z_opt = torch.zeros(
        (1, model.latent_dim),
        dtype=torch.float32,
        requires_grad=True,
        device=device,
    )
    optimizer = torch.optim.Adam([z_opt], lr=lr)
    losses = []

    for _ in range(n_steps):
        optimizer.zero_grad()
        recon, _, _ = model.decode(z_opt, device)
        recon_flat = recon[0, :2].reshape(2, -1)
        recon_observed = torch.cat(
            (recon_flat[0, mask_t], recon_flat[1, mask_t])
        )

        background_cost = 0.5 * torch.sum(z_opt**2)
        observation_residual = (recon_observed - target) / observation_error
        observation_cost = 0.5 * torch.sum(observation_residual**2)
        loss = background_cost + observation_cost
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    with torch.no_grad():
        _, final_cH, final_cP = model.decode(z_opt, device)
        G_t = torch.as_tensor(spline_matrix, dtype=torch.float32, device=device)
        H_log10 = final_cH @ G_t.T
        P_log10 = final_cP @ G_t.T
        H_phys = torch.clamp(
            torch.pow(10.0, H_log10) - epsilon,
            min=0.0,
        )
        P_phys = torch.clamp(
            torch.pow(10.0, P_log10) - epsilon,
            min=0.0,
        )

    return {
        "H": H_phys.reshape(36, 36).cpu().numpy(),
        "P": P_phys.reshape(36, 36).cpu().numpy(),
        "c_H": final_cH.detach().cpu().numpy(),
        "c_P": final_cP.detach().cpu().numpy(),
        "z": z_opt.detach().cpu().numpy(),
        "losses": np.asarray(losses),
    }

#%% 4. MAIN EXECUTION BLOCK

if __name__ == "__main__":
    
    print('Loading data')
    H, P, W, t, o, grid = helpers.load_data(PATHS, spar=CONFIG['SPARSITY_THRESHOLD'])

    fig, axs = plt.subplots(6, 6, figsize=(10, 10))
    ii = np.random.randint(0, 3664, 36)
    vmax = np.nanquantile(P, 0.99)
    for i, ax in zip(ii, axs.flatten()):
        ax.imshow(P[i], cmap='plasma', vmin=0, vmax=vmax)

    print('Define spline')
    spl = Spline(grid, 34, 3)
    G = spl.G
    ncp = spl.ncp

    print('Truncate data')
    subset_size = 10000
    H_trunc = H[:subset_size]
    P_trunc = P[:subset_size]
    W_trunc = W[:subset_size]

    print('Log of data')
    epsilon = 1
    log_H_trunc = np.log10(H_trunc + epsilon)
    log_P_trunc = np.log10(P_trunc + epsilon)

    print('Normalize data for VAE')
    H_norm, H_mu, H_std = normalize(log_H_trunc)
    P_norm, P_mu, P_std = normalize(log_P_trunc)

    print('Running VAE')
    model, losses, lrs = execute_VAE(H_norm, P_norm, W_trunc, beta=1)

    ####################################
    ### Probe latent space
    ####################################

    print('Probing latent space')
    c_H, c_P = probe_latent_space(model, CONFIG['LATENT_DIM'], n_samples=50000)
    
    log_H_samp, log_P_samp = c_H@G.T, c_P@G.T
    H_real = 10**(log_H_samp) - epsilon
    P_real = 10**(log_P_samp) - epsilon    
    #H_real, P_real = H_samp * H_std + H_mu, P_samp * P_std + P_mu
    
    print('Random samples of control points')
    rows, cols = 4, 6
    n = rows*cols
    np.random.seed(1)
    id_ = np.random.randint(0, c_H.shape[0], n)
    
    c_H_ = [c_H[i].reshape(ncp,ncp) for i in range(n)]
    c_P_ = [c_P[i].reshape(ncp,ncp) for i in range(n)]
    
    H_big = np.block([c_H_[i*cols:(i+1)*cols] for i in range(rows)])
    P_big = np.block([c_P_[i*cols:(i+1)*cols] for i in range(rows)])
    
    fig = plt.figure(figsize=(15, 10))
    plt.imshow(H_big, cmap='plasma', vmin=0, vmax=np.max(H_big), origin='lower')
    plt.title('24 random samples of Hall conductance control points')
    plt.gca().set_axis_off()
    
    fig = plt.figure(figsize=(15, 10))
    plt.imshow(P_big, cmap='plasma', vmin=0, vmax=np.max(P_big), origin='lower')
    plt.title('24 random samples of Pedersen conductance control points')
    plt.gca().set_axis_off()
    
    print('Random samples of conductance')
    H_real_ = [np.maximum(H_real[i], 1e-2).reshape(36,36) for i in range(n)]
    P_real_ = [np.maximum(P_real[i], 1e-2).reshape(36,36) for i in range(n)]
    
    H_big = np.block([H_real_[i*cols:(i+1)*cols] for i in range(rows)])
    P_big = np.block([P_real_[i*cols:(i+1)*cols] for i in range(rows)])
    
    fig = plt.figure(figsize=(15, 10))
    plt.imshow(H_big, cmap='plasma', vmin=0, vmax=np.max(H_big), origin='lower')
    plt.title('24 random samples of Hall conductance')
    plt.gca().set_axis_off()
    
    fig = plt.figure(figsize=(15, 10))
    plt.imshow(P_big, cmap='plasma', vmin=0, vmax=np.max(P_big), origin='lower')
    plt.title('24 random samples of Pedersen conductance')
    plt.gca().set_axis_off()

    ####################################
    ### Covariance - Control points
    ####################################
    
    print('Calculating covariance matrices')
    cov = calculate_joint_covariance_matrix(c_H, c_P)

    print('Plot teleconnections')
    n_control = ncp**2
    cov_HH = cov[:n_control, :n_control]
    cov_HP = cov[:n_control, n_control:]
    cov_PH = cov[n_control:, :n_control]
    cov_PP = cov[n_control:, n_control:]
    variance_H = np.diag(cov_HH)
    variance_P = np.diag(cov_PP)
    covariance_blocks = [
        ('H to H', cov_HH, variance_H, variance_H),
        ('H to P', cov_HP, variance_H, variance_P),
        ('P to P', cov_PP, variance_P, variance_P),
        ('P to H', cov_PH, variance_P, variance_H),
    ]
    
    fig, axs = plt.subplots(3, 4, figsize=(16, 12))
    points = [int(ncp/2*ncp+ncp/10), int(ncp/2*ncp+ncp/2), int(ncp/2*ncp+.8*ncp)]
    for i, p in enumerate(points):
        for ax, (title, cov_, row_variance, column_variance) in zip(
            axs[i], covariance_blocks
        ):
            tc = get_teleconnection(
                cov_,
                p,
                row_variances=row_variance,
                column_variances=column_variance,
            )
            ax.imshow(tc, cmap='bwr', vmin=-1, vmax=1)
            if i == 0:
                ax.set_title(title)
            ax.set_axis_off()

    print('Performing eigen decomposition')
    vecs, vals, vals_pct = perform_eigen_decomposition(cov)

    print('Comparing explained variance')
    plt.figure(figsize=(15, 9))
    c = np.cumsum(vals_pct)
    plt.plot(c)
    id_ = np.argmin(abs(c - 95))
    plt.plot([0, id_], [c[id_]]*2, color='tab:red')
    plt.plot([id_]*2, [0, c[id_]], color='tab:red')
    plt.xlabel('PC')
    plt.ylabel('Variance explained [%]')
    plt.title('joint covariance comparison')
    plt.legend()
    plt.xlim(-1, 250)
    plt.ylim(30, 101)
    plt.grid()

    fig, axs = illustrate_PCs(vecs[:ncp**2], vals_pct)
    plt.suptitle('PCA of joint covariance (Hall)')
    fig, axs = illustrate_PCs(vecs[ncp**2:], vals_pct)
    plt.suptitle('PCA of joint covariance (Pedersen)')

    ####################################
    ### Covariance - Conductance
    ####################################
    
    print('Calculating covariance matrices')
    cov = calculate_joint_covariance_matrix(H_real, P_real)

    print('Plot teleconnections')
    n_pixels = 36**2
    cov_HH = cov[:n_pixels, :n_pixels]
    cov_HP = cov[:n_pixels, n_pixels:]
    cov_PH = cov[n_pixels:, :n_pixels]
    cov_PP = cov[n_pixels:, n_pixels:]
    variance_H = np.diag(cov_HH)
    variance_P = np.diag(cov_PP)
    covariance_blocks = [
        ('H to H', cov_HH, variance_H, variance_H),
        ('H to P', cov_HP, variance_H, variance_P),
        ('P to P', cov_PP, variance_P, variance_P),
        ('P to H', cov_PH, variance_P, variance_H),
    ]
    
    fig, axs = plt.subplots(3, 4, figsize=(16, 12))
    points = [18*36+4, 18*36+18, 18*36+30]
    for i, p in enumerate(points):
        for ax, (title, cov_, row_variance, column_variance) in zip(
            axs[i], covariance_blocks
        ):
            tc = get_teleconnection(
                cov_,
                p,
                row_variances=row_variance,
                column_variances=column_variance,
            )
            ax.imshow(tc, cmap='bwr', vmin=-1, vmax=1)
            if i == 0:
                ax.set_title(title)
            ax.set_axis_off()

    print('Performing eigen decomposition')
    vecs, vals, vals_pct = perform_eigen_decomposition(cov)

    print('Comparing explained variance')
    plt.figure(figsize=(15, 9))
    c = np.cumsum(vals_pct)
    plt.plot(c)
    id_ = np.argmin(abs(c - 95))
    plt.plot([0, id_], [c[id_]]*2, color='tab:red')
    plt.plot([id_]*2, [0, c[id_]], color='tab:red')
    plt.xlabel('PC')
    plt.ylabel('Variance explained [%]')
    plt.title('joint covariance comparison')
    plt.legend()
    plt.xlim(-1, 250)
    plt.ylim(30, 101)
    plt.grid()

    fig, axs = illustrate_PCs(vecs[:36*36], vals_pct)
    plt.suptitle('PCA of joint covariance (Hall)')
    fig, axs = illustrate_PCs(vecs[36*36:], vals_pct)
    plt.suptitle('PCA of joint covariance (Pedersen)')


    ####################################
    ### Reconstruction
    ####################################
    
    idx = 3600
    H_test, P_test = H[idx].flatten(), P[idx].flatten()
    np.random.seed(2)
    drop_idx = np.unique(np.random.randint(0, 36*36, 2000)) # Heavy sparsification
    H_test[drop_idx] = np.nan
    P_test[drop_idx] = np.nan
    
    mask = ~(np.isnan(H_test) | np.isnan(P_test))
    H_sub, P_sub = H_test[mask], P_test[mask]
    
    log_H_sub = np.log10(H_sub + epsilon)
    log_P_sub = np.log10(P_sub + epsilon)
    
    rows = np.arange(36*36)[mask] // 36
    cols = np.arange(36*36)[mask] % 36
    
    vmin = 0
    vmax = np.nanmax(P_test)
    
    # Independent
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    for ax, tit in zip(axs.flatten(), 
                       ['Truth (Hall)', 'Input (Hall)', 'Reconstruction (Hall)', 'Truth (Ped)', 'Input (Ped)', 'Reconstruction (Ped)']):
        ax.set_axis_off()
        ax.set_title(tit)
    
    axs[0, 0].imshow(H[idx], cmap='plasma', origin='lower', vmin=vmin, vmax=vmax)
    axs[0, 0].scatter(cols, rows, 1, facecolor='k')
    axs[1, 0].imshow(P[idx], cmap='plasma', origin='lower', vmin=vmin, vmax=vmax)
    axs[1, 0].scatter(cols, rows, 1, facecolor='k')
    
    axs[0, 1].imshow(H_test.reshape(36, 36), cmap='plasma', origin='lower', vmin=vmin, vmax=vmax)
    axs[1, 1].imshow(P_test.reshape(36, 36), cmap='plasma', origin='lower', vmin=vmin, vmax=vmax)
    
    cov = calculate_joint_covariance_matrix(c_H, c_P)
    mean = np.mean(np.hstack((c_H, c_P)), axis=0)
    vals, vecs = scipy.linalg.eigh(cov)
    
    n_modes = 100
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx][:n_modes], vecs[:, idx][:, :n_modes]
    Q = (vecs * vals) @ vecs.T
    
    G_ = np.vstack((np.hstack((G, np.zeros(G.shape))), np.hstack((np.zeros(G.shape), G))))
    G_m = G_[np.hstack((mask, mask))]
    
    # Solve OI
    z = np.hstack((log_H_sub, log_P_sub))
    y = z - G_m@mean
    R = np.diag(np.ones(G_m.shape[0])*0.1**2)
    S = G_m@Q@G_m.T + R + np.eye(G_m.shape[0]) * 1e-6
    K = scipy.linalg.solve(S, (Q@G_m.T).T).T
    m = mean + K@y
    fit = G_@m
    log_H_fit = fit[:36**2]
    log_P_fit = fit[36**2:]
    
    H_fit = 10**log_H_fit - epsilon
    P_fit = 10**log_P_fit - epsilon
    
    axs[0, 2].imshow(H_fit.reshape(36, 36), cmap='plasma', origin='lower', vmin=vmin, vmax=vmax)
    axs[0, 2].scatter(cols, rows, 1, facecolor='k')
    axs[1, 2].imshow(P_fit.reshape(36, 36), cmap='plasma', origin='lower', vmin=vmin, vmax=vmax)
    axs[1, 2].scatter(cols, rows, 1, facecolor='k')
        
#%%    
    cov = calculate_joint_covariance_matrix(c_H, c_P)
    mean = np.mean(np.hstack((c_H, c_P)), axis=0)    
    vals, vecs = scipy.linalg.eigh(cov)    
    n_modes = 100
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx][:n_modes], vecs[:, idx][:, :n_modes]
    Q = (vecs * vals) @ vecs.T
    G_ = np.vstack((np.hstack((G, np.zeros(G.shape))), np.hstack((np.zeros(G.shape), G))))

    H_true, P_true, H_pred, P_pred = [], [], [], []
    cols, rows, vmax = [], [], []
    for i, idx in enumerate([3010, 3070, 3080, 3090, 3140, 3600]):
        H_test, P_test = H[idx].flatten(), P[idx].flatten()
        
        H_true.append(copy.deepcopy(H_test.reshape(36,36)))
        P_true.append(copy.deepcopy(P_test.reshape(36,36)))
        
        np.random.seed(i)
        drop_idx = np.unique(np.random.randint(0, 36*36, 4000)) # Heavy sparsification
        H_test[drop_idx] = np.nan
        P_test[drop_idx] = np.nan
    
        mask = ~(np.isnan(H_test) | np.isnan(P_test))
        H_sub, P_sub = H_test[mask], P_test[mask]
        
        log_H_sub = np.log10(H_sub + epsilon)
        log_P_sub = np.log10(P_sub + epsilon)
    
        rows.append(np.arange(36*36)[mask] // 36)
        cols.append(np.arange(36*36)[mask] % 36)        
    
        vmax.append(np.nanmax(P_test))
            
        G_m = G_[np.hstack((mask, mask))]
        
        z = np.hstack((log_H_sub, log_P_sub))
        y = z - G_m@mean
        R = np.diag(np.ones(G_m.shape[0])*0.1**2)
        S = G_m@Q@G_m.T + R #+ np.eye(G_m.shape[0]) * 1e-6
        K = scipy.linalg.solve(S, (Q@G_m.T).T).T
        m = mean + K@y
        fit = G_@m
        log_H_fit = fit[:36**2]
        log_P_fit = fit[36**2:]
        
        H_fit = 10**log_H_fit - epsilon
        P_fit = 10**log_P_fit - epsilon
        
        H_pred.append(H_fit.reshape(36,36))
        P_pred.append(P_fit.reshape(36,36))
    
    dcl = 'cyan'
    fig, axs = plt.subplots(4, 6, figsize=(15, 10))
    for i in range(6):
        axs[0, i].imshow(H_true[i], cmap='plasma', origin='lower', vmin=0, vmax=vmax[i])
        axs[1, i].imshow(H_pred[i], cmap='plasma', origin='lower', vmin=0, vmax=vmax[i])
        axs[2, i].imshow(P_true[i], cmap='plasma', origin='lower', vmin=0, vmax=vmax[i])
        axs[3, i].imshow(P_pred[i], cmap='plasma', origin='lower', vmin=0, vmax=vmax[i])
        axs[0, i].scatter(cols[i], rows[i], 1, facecolor=dcl)
        axs[1, i].scatter(cols[i], rows[i], 1, facecolor=dcl)
        axs[2, i].scatter(cols[i], rows[i], 1, facecolor=dcl)
        axs[3, i].scatter(cols[i], rows[i], 1, facecolor=dcl)
    for ax in axs.flatten():
        ax.set_axis_off()
    axs[0,0].text(-.1, .5, 'Truth (Hall)', ha='center', va='center', rotation='vertical', transform=axs[0,0].transAxes)
    axs[1,0].text(-.1, .5, 'Recon (Hall)', ha='center', va='center', rotation='vertical', transform=axs[1,0].transAxes)
    axs[2,0].text(-.1, .5, 'Truth (Pedersen)', ha='center', va='center', rotation='vertical', transform=axs[2,0].transAxes)
    axs[3,0].text(-.1, .5, 'Recon (Pedersen)', ha='center', va='center', rotation='vertical', transform=axs[3,0].transAxes)
    for i, ax in enumerate(axs[0, :]):
        ax.text(.5, 1.1, f'Example {i}', ha='center', va='center', transform=ax.transAxes)
    
    
#%% Latent-space assimilation in normalized log10 space
    
    H_true, P_true, H_pred, P_pred = [], [], [], []
    cols, rows, vmax = [], [], []
    for i, idx in enumerate([3010, 3070, 3080, 3090, 3140, 3600]):
        H_test, P_test = H[idx].flatten(), P[idx].flatten()
        
        H_true.append(copy.deepcopy(H_test.reshape(36,36)))
        P_true.append(copy.deepcopy(P_test.reshape(36,36)))
        
        np.random.seed(i)
        drop_idx = np.unique(np.random.randint(0, 36*36, 4000)) # Heavy sparsification
        H_test[drop_idx] = np.nan
        P_test[drop_idx] = np.nan
    
        mask = ~(np.isnan(H_test) | np.isnan(P_test))
        H_sub, P_sub = H_test[mask], P_test[mask]
    
        rows.append(np.arange(36*36)[mask] // 36)
        cols.append(np.arange(36*36)[mask] % 36)        
    
        vmax.append(np.nanmax(P_test))
                
        assimilation = latent_space_assimilation(
            model,
            H_sub,
            P_sub,
            mask,
            G,
            (H_mu, H_std),
            (P_mu, P_std),
            observation_error_phys=0.3,
            epsilon=epsilon,
            n_steps=10000,
            lr=1e-4,
        )
    
        H_pred.append(assimilation["H"])
        P_pred.append(assimilation["P"])
    
    dcl = 'cyan'
    fig, axs = plt.subplots(4, 6, figsize=(15, 10))
    for i in range(6):
        axs[0, i].imshow(H_true[i], cmap='plasma', origin='lower', vmin=0, vmax=vmax[i])
        axs[1, i].imshow(H_pred[i], cmap='plasma', origin='lower', vmin=0, vmax=vmax[i])
        axs[2, i].imshow(P_true[i], cmap='plasma', origin='lower', vmin=0, vmax=vmax[i])
        axs[3, i].imshow(P_pred[i], cmap='plasma', origin='lower', vmin=0, vmax=vmax[i])
        axs[0, i].scatter(cols[i], rows[i], 1, facecolor=dcl)
        axs[1, i].scatter(cols[i], rows[i], 1, facecolor=dcl)
        axs[2, i].scatter(cols[i], rows[i], 1, facecolor=dcl)
        axs[3, i].scatter(cols[i], rows[i], 1, facecolor=dcl)
    for ax in axs.flatten():
        ax.set_axis_off()
    axs[0,0].text(-.1, .5, 'Truth (Hall)', ha='center', va='center', rotation='vertical', transform=axs[0,0].transAxes)
    axs[1,0].text(-.1, .5, 'Recon (Hall)', ha='center', va='center', rotation='vertical', transform=axs[1,0].transAxes)
    axs[2,0].text(-.1, .5, 'Truth (Pedersen)', ha='center', va='center', rotation='vertical', transform=axs[2,0].transAxes)
    axs[3,0].text(-.1, .5, 'Recon (Pedersen)', ha='center', va='center', rotation='vertical', transform=axs[3,0].transAxes)
    for i, ax in enumerate(axs[0, :]):
        ax.text(.5, 1.1, f'Example {i}', ha='center', va='center', transform=ax.transAxes)
