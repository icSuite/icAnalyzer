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
        #return torch.sigmoid(x)

    def forward(self, x, mask):
        mu, logvar = self.encode(x, mask)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def loss_function_old(recon, x, mask, mu, logvar, beta=1):
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

def loss_function(recon, x, mask, mu, logvar, beta=1):
    
    # 1. Standard MSE on normalized data (same as before)
    recon_target = recon[:, :2, :, :] 
    target = x[:, :2, :, :]
    weights = x[:, 2:3, :, :] * mask
    mse_loss = torch.sum(weights * (recon_target - target)**2) / (weights.sum() + 1e-8) * (36*36*2)

    # 2. THE PHYSICS PENALTY (Asymmetric Loss)
    # Convert predictions back to physical space conceptually inside the loss
    H_phys_pred = (recon[:, 0, :, :] * H_std) + H_mu
    P_phys_pred = (recon[:, 1, :, :] * P_std) + P_mu
    
    # If physical prediction is < 0, penalize it heavily. If > 0, penalty is 0.
    # ReLU turns negative numbers into 0, so we use ReLU(-prediction) to isolate the negatives.
    neg_penalty_H = torch.sum(F.relu(-H_phys_pred)**2) 
    neg_penalty_P = torch.sum(F.relu(-P_phys_pred)**2)
    
    physics_loss = 10.0 * (neg_penalty_H + neg_penalty_P) # Multiply by a strong weight

    # 3. KLD
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return mse_loss + physics_loss + (beta * kld_loss)

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
            loss = loss_function(recon, data, mask, mu, logvar)
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

def execute_VAE_old(H, P, W):
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
    losses = []
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
        losses.append(total_loss)
            
        print(f"Epoch {epoch+1}/{CONFIG['EPOCHS']} | Avg Loss: {total_loss/len(dataset):.4f}")
    
    return model, losses

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

def get_teleconnection(cov_matrix, base_pixel_idx):
    # 1. Extract the diagonal (variances)
    variances = np.diag(cov_matrix)
    
    # 2. Convert Covariance to Correlation: R = Cov(X,Y) / (Std(X) * Std(Y))
    std_devs = np.sqrt(variances)
    correlation_matrix = cov_matrix / np.outer(std_devs, std_devs)
    
    # 3. Pull the specific row for your base pixel
    teleconnections = correlation_matrix[base_pixel_idx, :]
    
    return teleconnections.reshape(36, 36) # Plot this with vmin=-1, vmax=1, cmap='RdBu_r'

def calculate_reconstruction_errors(model, H_full, P_full, W_full, H_stats, P_stats, batch_size=32):
    """
    Passes the full dataset through the trained VAE and calculates the 
    Mean Squared Error (MSE) for each individual image.
    """
    device = next(model.parameters()).device
    model.eval() # CRITICAL: Turn off dropout and batchnorm updates
    
    H_mu, H_std = H_stats
    P_mu, P_std = P_stats
    
    print("Preparing full dataset...")
    # 1. Normalize using the EXACT SAME stats from your clean training set
    H_norm = (H_full - H_mu) / H_std
    P_norm = (P_full - P_mu) / P_std
    
    # 2. Create Masks and Stack
    mask_np = (~(np.isnan(H_full) | np.isnan(P_full) | np.isnan(W_full))).astype(np.float32)
    X_np = np.stack([np.nan_to_num(H_norm, nan=0.0), 
                     np.nan_to_num(P_norm, nan=0.0),
                     np.nan_to_num(W_full, nan=0.0)], axis=1)
    
    tensor_x = torch.from_numpy(X_np).float()
    tensor_mask = torch.from_numpy(mask_np).float().unsqueeze(1)
    
    # 3. DataLoader (shuffle=False is mandatory here!)
    dataset = TensorDataset(tensor_x, tensor_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_errors = []
    
    print("Calculating reconstruction errors...")
    with torch.no_grad(): # Don't track gradients, saves massive amounts of memory
        for data, mask in tqdm(dataloader):
            data, mask = data.to(device), mask.to(device)
            
            # Forward pass
            recon, _, _ = model(data, mask)
            
            # Extract targets and predictions
            target = data[:, :2, :, :]       # (B, 2, 36, 36)
            recon_target = recon[:, :2, :, :] # (B, 2, 36, 36)
            
            # Combine Weight and Mask
            weight_mask = data[:, 2:3, :, :] * mask # (B, 1, 36, 36)
            
            # Calculate Squared Error per pixel
            squared_error = (recon_target - target)**2
            weighted_sq_error = squared_error * weight_mask
            
            # Sum over Channels (dim=1), Height (dim=2), Width (dim=3)
            # This gives us a 1D array of length B (one error value per image)
            sum_error_per_image = torch.sum(weighted_sq_error, dim=(1, 2, 3))
            
            # Divide by the sum of the weights to get the MEAN error per valid pixel
            # Multiply by 2 because we have 2 channels (H and P)
            valid_weight_sum = torch.sum(weight_mask, dim=(1, 2, 3)) * 2 
            
            mean_error_per_image = sum_error_per_image / (valid_weight_sum + 1e-8)
            
            # Store results
            all_errors.extend(mean_error_per_image.cpu().numpy())
            
    return np.array(all_errors)

#%% 4. MAIN EXECUTION BLOCK

if __name__ == "__main__":
    
    print('Loading data')
    H, P, W, t, o, grid = helpers.load_data(PATHS, spar=CONFIG['SPARSITY_THRESHOLD'])

    print('Truncate data')
    subset_size = 3000
    H_trunc = H[:subset_size]
    P_trunc = P[:subset_size]
    W_trunc = W[:subset_size]

    print('Normalize data for VAE')
    H_norm, H_mu, H_std = normalize(H_trunc)
    P_norm, P_mu, P_std = normalize(P_trunc)

    print('Running VAE')
    model, losses, lrs = execute_VAE(H_norm, P_norm, W_trunc)

    ####################################
    ### Probe latent space
    ####################################

    print('Probing latent space')
    H_samp, P_samp, W_samp = probe_latent_space(model, CONFIG['LATENT_DIM'], n_samples=50000)

    H_real, P_real = H_samp * H_std + H_mu, P_samp * P_std + P_mu

    rows, cols = 4, 6
    n = rows*cols
    np.random.seed(1)
    id_ = np.random.randint(0, H_real.shape[0], n)
    
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
    
    fig = plt.figure(figsize=(15, 10))
    plt.imshow(np.log(P_big), cmap='plasma', vmin=np.min(np.log(P_big)), vmax=np.max(np.log(P_big)), origin='lower')
    plt.title('24 random samples of Pedersen conductance (log-space)')
    plt.gca().set_axis_off()

    ####################################
    ### Linear covariance
    ####################################
    
    print('Calculating covariance matrices (linear)')
    cov_H = calculate_covariance_matrix(H_samp)
    cov_P = calculate_covariance_matrix(P_samp)
    cov   = calculate_joint_covariance_matrix(H_samp, P_samp)

    print('Plot teleconnections')
    cov_HH, cov_HP = cov[:36*36, :36*36], cov[36*36:, :36*36]
    cov_PP, cov_PH = cov[36*36:, 36*36:], cov[:36*36, 36*36:]
    
    fig, axs = plt.subplots(3, 6, figsize=(16, 8))
    points = [18*36+4, 18*36+18, 18*36+30]
    for i, p in enumerate(points):
        for ax, cov_ in zip(axs[i], [cov_H, cov_P, cov_HH, cov_HP, cov_PP, cov_PH]):
            tc = get_teleconnection(cov_, p)
            ax.imshow(tc, cmap='bwr', vmin=-1, vmax=1)

    print('Performing eigen decomposition')
    H_vecs, H_vals, H_vals_pct = perform_eigen_decomposition(cov_H)
    P_vecs, P_vals, P_vals_pct = perform_eigen_decomposition(cov_P)
    vecs, vals, vals_pct       = perform_eigen_decomposition(cov)

    print('Comparing explained variance')
    plt.figure(figsize=(15, 9))
    for x, lab in zip([H_vals_pct, P_vals_pct, vals_pct], ['Hall', 'Pedersen', 'Joint']):
        c = np.cumsum(x)
        plt.plot(c, label=lab)
        id_ = np.argmin(abs(c - 95))
        plt.plot([0, id_], [c[id_]]*2, color='tab:red')
        plt.plot([id_]*2, [0, c[id_]], color='tab:red')
    
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


    ####################################
    ### Log covariance
    ####################################
    
    print('Calculate log of samples')
    H_samp_log = np.maximum((H_samp * H_std) + H_mu, 0)
    P_samp_log = np.maximum((P_samp * P_std) + P_mu, 0)
    
    eps = 1 # hmm
    H_samp_log = np.log(H_samp_log + eps)
    P_samp_log = np.log(P_samp_log + eps)
    
    print('Calculating average')
    H_log_mu = np.mean(H_samp_log, axis=0)
    P_log_mu = np.mean(P_samp_log, axis=0)
    
    print('Calculating covariance matrices')    
    cov_H = calculate_covariance_matrix(H_samp_log)
    cov_P = calculate_covariance_matrix(P_samp_log)
    cov   = calculate_joint_covariance_matrix(H_samp_log, P_samp_log)

    print('Plot teleconnections')
    cov_HH, cov_HP = cov[:36*36, :36*36], cov[36*36:, :36*36]
    cov_PP, cov_PH = cov[36*36:, 36*36:], cov[:36*36, 36*36:]
    
    fig, axs = plt.subplots(3, 6, figsize=(16, 8))
    points = [18*36+4, 18*36+18, 18*36+30]
    for i, p in enumerate(points):
        for ax, cov_ in zip(axs[i], [cov_H, cov_P, cov_HH, cov_HP, cov_PP, cov_PH]):
            tc = get_teleconnection(cov_, p)
            ax.imshow(tc, cmap='bwr', vmin=-1, vmax=1)

    print('Performing eigen decomposition')
    H_vecs, H_vals, H_vals_pct = perform_eigen_decomposition(cov_H)
    P_vecs, P_vals, P_vals_pct = perform_eigen_decomposition(cov_P)
    vecs, vals, vals_pct       = perform_eigen_decomposition(cov)

    print('Comparing explained variance')
    plt.figure(figsize=(15, 9))
    for x, lab in zip([H_vals_pct, P_vals_pct, vals_pct], ['Hall', 'Pedersen', 'Joint']):
        c = np.cumsum(x)
        plt.plot(c, label=lab)
        id_ = np.argmin(abs(c - 95))
        plt.plot([0, id_], [c[id_]]*2, color='tab:red')
        plt.plot([id_]*2, [0, c[id_]], color='tab:red')
    
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
    np.random.seed(2)
    drop_idx = np.unique(np.random.randint(0, 36*36, 4000)) # Heavy sparsification
    H_test[drop_idx] = np.nan
    P_test[drop_idx] = np.nan
    
    mask = ~(np.isnan(H_test) | np.isnan(P_test))
    H_sub, P_sub = H_test[mask], P_test[mask]
    
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
    
    P_sub_log_std = .3 / (P_sub + eps)
    P_sub_log = np.log(P_sub + eps)
    P_fit = log_OI(P_log_mu, cov_P, P_sub_log, P_sub_log_std, mask, n_modes=500)
    H_sub_log_std = .3 / (H_sub + eps)
    H_sub_log = np.log(H_sub + eps)
    H_fit = log_OI(H_log_mu, cov_H, H_sub_log, H_sub_log_std, mask, n_modes=500)
    
    axs[0, 2].imshow(H_fit.reshape(36, 36), cmap='plasma', origin='lower', vmin=vmin, vmax=vmax)
    axs[0, 2].scatter(cols, rows, 1, facecolor='k')
    axs[1, 2].imshow(P_fit.reshape(36, 36), cmap='plasma', origin='lower', vmin=vmin, vmax=vmax)
    axs[1, 2].scatter(cols, rows, 1, facecolor='k')
    
    
    
    
    # Joint
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

    x_sub = np.hstack((H_sub, P_sub))
    x_sub_log_std = .2 / (x_sub + eps)
    x_sub_log = np.log(x_sub + eps)
    x_fit = log_OI(np.hstack((H_log_mu, P_log_mu)), cov, x_sub_log, x_sub_log_std, np.hstack((mask, mask)), n_modes=250)
    
    axs[0, 2].imshow(x_fit[:36*36].reshape(36, 36), cmap='plasma', origin='lower', vmin=vmin, vmax=vmax)
    axs[0, 2].scatter(cols, rows, 1, facecolor='k')
    axs[1, 2].imshow(x_fit[36*36:].reshape(36, 36), cmap='plasma', origin='lower', vmin=vmin, vmax=vmax)
    axs[1, 2].scatter(cols, rows, 1, facecolor='k')
    
    
    
#%%    
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
    
        std_ = .2
        n_modes = 500
        x_sub = np.hstack((H_sub, P_sub))
        x_sub_log_std = std_ / (x_sub + eps)
        x_sub_log = np.log(x_sub + eps)
        x_fit = log_OI(np.hstack((H_log_mu, P_log_mu)), cov, x_sub_log, x_sub_log_std, np.hstack((mask, mask)), n_modes=n_modes)
    
        H_pred.append(x_fit[:36*36].reshape(36,36))
        P_pred.append(x_fit[36*36:].reshape(36,36))
    
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
    

