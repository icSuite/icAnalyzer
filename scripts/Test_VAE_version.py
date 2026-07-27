#%% Import
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import glob
from icreader import ConductanceImage # Commented out for portability, un-comment in your env
from tqdm import tqdm

from os.path import join as pjoin
import pandas as pd
import matplotlib.pyplot as plt

#%% Fun

# ==========================================
# 1. HELPER CLASS: Partial Convolution
# ==========================================
class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        # Standard Conv2d initialization
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False
        
        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False
            
        super(PartialConv2d, self).__init__(*args, **kwargs)

        # Fixed kernel of 1s for mask updating
        # Shape: (Out=1, In=1, H, W)
        self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
        self.slide_winsize = self.kernel_size[0] * self.kernel_size[1]
        
        self.last_size = (0, 0, 0, 0)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        # input: (B, C, H, W)
        # mask_in: (B, C_mask, H, W)
        
        # --- FIX: Ensure mask is always 1-channel for broadcasting ---
        if mask_in is not None and mask_in.shape[1] > 1:
            # We assume the mask is identical across channels (Hall/Ped),
            # so we just take the first channel.
            mask_in = mask_in[:, 0:1, :, :]
            
        with torch.no_grad():
            if self.weight_maskUpdater.type() != input.type():
                self.weight_maskUpdater = self.weight_maskUpdater.to(input)

            if mask_in is None:
                # If no mask provided, assume all valid (1 channel)
                mask_in = torch.ones(input.size(0), 1, input.size(2), input.size(3)).to(input)

            # Update Mask: Count valid pixels in each window
            # Input is (B, 1, H, W), Weight is (1, 1, K, K) -> Output (B, 1, H, W)
            update_mask = F.conv2d(mask_in, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

            # Calculate scaling factor: (Total Pixels / Valid Pixels)
            self.mask_ratio = self.slide_winsize / (update_mask + 1e-8)
            self.update_mask = torch.clamp(update_mask, 0, 1)
            self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        # Standard Convolution
        # mask_in (B, 1, H, W) broadcasts to input (B, C, H, W) automatically
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

# ==========================================
# 2. DROP-IN REPLACEMENT: VAE2D
# ==========================================
class VAE2D(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE2D, self).__init__()
        self.latent_dim = latent_dim

        # --- ENCODER (Partial Conv) ---
        # 1. 36 -> 18
        self.enc_pconv1 = PartialConv2d(2, 16, kernel_size=3, stride=2, padding=1, return_mask=True)
        self.enc_bn1 = nn.BatchNorm2d(16)
        
        # 2. 18 -> 9
        self.enc_pconv2 = PartialConv2d(16, 32, kernel_size=3, stride=2, padding=1, return_mask=True)
        self.enc_bn2 = nn.BatchNorm2d(32)
        
        # 3. 9 -> 5
        self.enc_pconv3 = PartialConv2d(32, 64, kernel_size=3, stride=2, padding=1, return_mask=True)
        self.enc_bn3 = nn.BatchNorm2d(64)
        
        self.flatten_dim = 64 * 5 * 5 
        
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # --- DECODER (Standard Conv) ---
        # Since latent space Z is dense (no holes), we don't need PConv here.
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        
        # 1. (64, 5, 5) -> (32, 9, 9)
        self.dec_conv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.dec_bn1 = nn.BatchNorm2d(32)
        
        # 2. (32, 9, 9) -> (16, 18, 18)
        self.dec_conv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn2 = nn.BatchNorm2d(16)
        
        # 3. (16, 18, 18) -> (2, 36, 36)
        # Note: We output 2 channels (Hall, Pedersen)
        self.dec_conv3 = nn.ConvTranspose2d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=1)

    def encode(self, x, mask):
        # Layer 1: Pass x and mask, get new features and new mask
        x1, m1 = self.enc_pconv1(x, mask_in=mask)
        x1 = F.relu(self.enc_bn1(x1))
        
        # Layer 2: Use m1
        x2, m2 = self.enc_pconv2(x1, mask_in=m1)
        x2 = F.relu(self.enc_bn2(x2))
        
        # Layer 3: Use m2
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
        # IMPORTANT: You must pass 'mask' to forward now
        mu, logvar = self.encode(x, mask)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# ==========================================
# 3. UPDATED LOSS FUNCTION (Weighted)
# ==========================================
def loss_function(recon, x, mask, mu, logvar, beta=1):
    # Calculate MSE only on valid pixels (using the original mask)
    # We weight valid pixels by 1, invalid by 0
    recon_loss = F.mse_loss(recon * mask, x * mask, reduction='sum')
    
    # Normalize by number of valid pixels to prevent loss magnitude
    # from fluctuating based on how much data is missing
    valid_count = mask.sum() + 1e-8
    recon_loss = recon_loss / valid_count * (x.shape[2] * x.shape[3])
    
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kld_loss

#%% Hyper paramaters

CHANNELS = 2           # Hall, Hall_Unc, Pedersen, Pedersen_Unc, Mask
IMG_HEIGHT = 36        
IMG_WIDTH = 36         
LATENT_DIM = 36        
BATCH_SIZE = 8
LEARNING_RATE = 1e-4

#%% Paths

base = '/home/bing/Dropbox/work'
p_in = pjoin(base, 'data', 'IMAGE_FUV', 'conductance')
p_out = pjoin(base, 'code', 'repos', 'icAnalyzer', 'figures', 'overview_wic')

#%% Load good segments of data

filename = pjoin(base, 'code', 'repos', 'icAnalyzer', 'data', 'good_orbit_segments.csv')
gos = pd.read_csv(filename, header=None)
gos.columns = ['orbit', 'shit_start', 'shit_stop', 'good_start', 'good_stop']
gos = gos.loc[gos['orbit']<503, :]

t, H, P, o = [], [], [], []
for orbit in tqdm(gos['orbit'], total=gos.shape[0]):
    cI = ConductanceImage(pjoin(p_in, f'or_{str(orbit).zfill(4)}.nc'))
    
    start = gos.loc[gos['orbit']==orbit].iloc[0]['good_start']
    stop = gos.loc[gos['orbit']==orbit].iloc[0]['good_stop']
    
    if stop == 0:
        continue
    if np.isnan(start):
        start = 0
    if np.isnan(stop):
        stop = cI.H.shape[0]-1
    
    start = int(start)
    stop = int(stop+1)
    H.append(cI.H[start:stop])
    P.append(cI.P[start:stop])
    t.append(cI.time[start:stop])
    o.append(np.ones(cI.time[start:stop].size)*orbit)

H = np.concatenate(H, axis=0)
P = np.concatenate(P, axis=0)
t = np.concatenate(t)
o = np.concatenate(o)

print(np.min(t))
print(np.max(t))

#%%

h, w = 36, 36
f = np.sum((np.isnan(H) | np.isnan(P)), axis=(1,2)) / (h*w)
f = f < .33
H, P = H[f], P[f]

#%%

H_mu, H_sig = np.nanmean(H), np.nanstd(H)
P_mu, P_sig = np.nanmean(P), np.nanstd(P)

H_norm = (H-H_mu)/H_sig
P_norm = (P-P_mu)/P_sig

H_trunc = H_norm[:10000]
P_trunc = P_norm[:10000]

# Create 1-Channel Mask (N, 36, 36)
mask_np = (~(np.isnan(H_trunc) | np.isnan(P_trunc))).astype(np.float32)

# Input Data: (N, 2, 36, 36)
X_np = np.stack([np.nan_to_num(H_trunc, nan=0.0), np.nan_to_num(P_trunc, nan=0.0)], axis=1)
tensor_x = torch.from_numpy(X_np).float()

# Mask Data: (N, 1, 36, 36) <-- CHANGED from stack to unsqueeze
tensor_mask = torch.from_numpy(mask_np).float().unsqueeze(1)

#%%

# ==========================================
# 1. PREPARE DATASET (Combine Data + Mask)
# ==========================================
# tensor_x: (N, 2, 36, 36)
# tensor_mask: (N, 2, 36, 36)

# Create a dataset that yields pairs of (image, mask)
dataset = TensorDataset(tensor_x, tensor_mask)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==========================================
# 2. INITIALIZE MODEL
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE2D(latent_dim=LATENT_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Model initialized on {device}")

# ==========================================
# 3. UPDATED TRAINING LOOP
# ==========================================
num_epochs = 100 

model.train()
for epoch in range(num_epochs):
    total_loss = 0
    
    # Unpack the batch: 'data' is image, 'mask' is the binary mask
    for batch_idx, (data, mask) in enumerate(dataloader):
        data = data.to(device)
        mask = mask.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass: Pass BOTH data and mask to the PartialConv model
        # The model uses the mask to ignore 0s in the convolution
        recon_batch, mu, logvar = model(data, mask)
        
        # Loss: Pass mask to ensure we only calculate error on valid pixels
        loss = loss_function(recon_batch, data, mask, mu, logvar)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader.dataset)
    print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')


#%%

recon, mu, logvar = model(tensor_x, tensor_mask)

with torch.no_grad():
    fig, axs = plt.subplots(2, 5)
    for i in range(5):
        axs[0, i].imshow(tensor_x[i*50, 0], cmap='plasma')
        axs[1, i].imshow(recon[i*50, 0], cmap='plasma')

#%%

id_ = 3010
np.random.seed(1)

H_, P_ = H_norm[id_].flatten(), P_norm[id_].flatten()
idx = np.unique(np.random.randint(0, 36*36-1, 3000))
H_[idx], P_[idx] = np.nan, np.nan

mask = (~(np.isnan(H_) | np.isnan(P_))).astype(np.float32).reshape((36, 36))
H_ = np.nan_to_num(H_, nan=0.0).reshape((36, 36))
P_ = np.nan_to_num(P_, nan=0.0).reshape((36, 36))

tX_norm = torch.from_numpy(np.stack([H_, P_], axis=0)).float().unsqueeze(0)
tmask = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
recon, mu, logvar = model(tX_norm, tmask)

x_ = (P_norm[id_] + P_mu)*P_sig
x_nan = (tX_norm[0, 1] + P_mu)*P_sig
with torch.no_grad():
    pred = (recon[0, 1]+P_mu)*P_sig

fig, axs = plt.subplots(2, 2)

ax = axs[0, 0]
im = ax.imshow(x_, cmap='plasma', vmin=0, vmax=np.nanmax(x_))
fig.colorbar(im, ax=ax)

ax = axs[0, 1]
im = ax.imshow(x_nan, cmap='plasma', vmin=0, vmax=np.nanmax(x_))
fig.colorbar(im, ax=ax)

ax = axs[1, 0]
im = ax.imshow(pred, cmap='plasma', vmin=0, vmax=np.nanmax(x_))
fig.colorbar(im, ax=ax)

ax = axs[1, 1]
im = ax.imshow(pred - x_, cmap='bwr')
fig.colorbar(im, ax=ax)

#%%

# ==========================================
# 1. LATENT OPTIMIZATION FUNCTION
# ==========================================
def reconstruct_from_sparse(model, sparse_data, sparse_mask, iterations=200, lr=0.1):
    """
    Optimizes latent vector z to match the sparse measurements.
    
    Args:
        model: Trained VAE2D model
        sparse_data: Tensor (1, 2, 36, 36) - Hall/Pedersen with 0s at gaps
        sparse_mask: Tensor (1, 1, 36, 36) - 1 at data, 0 at gaps
        iterations: Number of optimization steps
    """
    # Ensure inputs are on the correct device
    device = next(model.parameters()).device
    sparse_data = sparse_data.to(device)
    sparse_mask = sparse_mask.to(device)
    
    # 1. Start with a random latent vector z (Leaf Tensor)
    # We use the model's latent_dim to ensure size match
    z = torch.randn(1, model.latent_dim, device=device, requires_grad=True)
    
    # 2. Define Optimizer for z
    optimizer = torch.optim.Adam([z], lr=lr)
    
    model.eval() # Important: turn off batchnorm/dropout updates
    
    # 3. Optimization Loop
    for i in range(iterations):
        optimizer.zero_grad()
        
        # Decode current z
        recon = model.decode(z) # Output: (1, 2, 36, 36)
        
        # Calculate loss ONLY at valid measurement locations
        # We broadcast the (1,1,36,36) mask to (1,2,36,36) to cover both channels
        loss = F.mse_loss(recon * sparse_mask, sparse_data * sparse_mask, reduction='sum')
        
        # Add a small regularization term to keep z close to normal distribution (Prior)
        # This prevents z from exploding to unrealistic values
        loss = loss + 0.01 * torch.sum(z ** 2)
        
        loss.backward()
        optimizer.step()
        
        # Optional: Print progress
        # if i % 50 == 0:
        #     print(f"Iter {i}: Loss {loss.item():.4f}")
            
    # Return the final reconstruction
    return model.decode(z).detach().cpu().numpy()

# ==========================================
# 2. DATA PREP & EXECUTION
# ==========================================
id_ = 3010
np.random.seed(1) # Fixed seed for reproducibility

# Get a sample
# Flatten to easily select random indices
H_flat, P_flat = H_norm[id_].flatten(), P_norm[id_].flatten()

# Simulate Gaps: Create indices to DROP
# 36*36 = 1296 pixels. Dropping 1200 leaves ~96 points (Sparse!)
drop_idx = np.unique(np.random.randint(0, 36*36, 3000))

H_flat[drop_idx] = np.nan
P_flat[drop_idx] = np.nan

# Reshape back to image
H_sparse = H_flat.reshape((36, 36))
P_sparse = P_flat.reshape((36, 36))

# Create Mask (1-channel)
# 1 where data exists, 0 where nan
mask_np = (~(np.isnan(H_sparse) | np.isnan(P_sparse))).astype(np.float32)

# Create Data (Fill NaNs with 0)
H_filled = np.nan_to_num(H_sparse, nan=0.0)
P_filled = np.nan_to_num(P_sparse, nan=0.0)

# Convert to Tensors
# Data: (1, 2, 36, 36)
sparse_data_tensor = torch.from_numpy(np.stack([H_filled, P_filled], axis=0)).float().unsqueeze(0)
# Mask: (1, 1, 36, 36)
sparse_mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)

print(f"Optimizing Latent Vector for {sparse_mask_tensor.sum()} valid points...")

# RUN RECONSTRUCTION
full_map = reconstruct_from_sparse(model, sparse_data_tensor, sparse_mask_tensor, iterations=300)

# ==========================================
# 3. PLOTTING
# ==========================================
# Original Ground Truth (for comparison)
# Note: Re-using your scaling logic
original_P = (P_norm[id_] + P_mu) * P_sig

# Sparse Input (Visualization only)
input_P = (sparse_data_tensor[0, 1].numpy() + P_mu) * P_sig
#input_P[mask_np == 0] = np.nan # Set gaps to nan for plotting

# Reconstructed Prediction
pred_P = (full_map[0, 1] + P_mu) * P_sig

# Plot
fig, axs = plt.subplots(2, 2)

# 1. Ground Truth
ax = axs[0, 0]
im = ax.imshow(original_P, cmap='plasma', vmin=0, vmax=np.nanmax(original_P))
fig.colorbar(im, ax=ax)

# 2. Sparse Input
ax = axs[0, 1]
im = ax.imshow(input_P, cmap='plasma', vmin=0, vmax=np.nanmax(original_P))
fig.colorbar(im, ax=ax)

# 3. Latent Reconstruction
ax = axs[1, 0]
im = ax.imshow(pred_P, cmap='plasma', vmin=0, vmax=np.nanmax(original_P))
fig.colorbar(im, ax=ax)

# 4. Error (Pred - Truth)
ax = axs[1, 1]
diff = pred_P - original_P
im = ax.imshow(diff, cmap='bwr')
fig.colorbar(im, ax=ax)

#%%

def generate_vae_covariance(model, latent_dim, channel_idx=0, n_samples=5000):
    """
    Generates a Covariance Matrix by probing the VAE latent space.
    
    Args:
        channel_idx: 0 for Hall, 1 for Pedersen
        n_samples: Higher = smoother, more converged matrix (5000+ recommended)
    """
    # Detect device from model parameters
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        # 1. Sample from the Latent Prior (Standard Normal)
        z = torch.randn(n_samples, latent_dim).to(device)
        
        # 2. Decode into Images
        # Output shape: (n_samples, CHANNELS, 36, 36)
        samples = model.decode(z)
        
        # 3. Focus on a specific channel and flatten spatial dims
        # Shape: (n_samples, 1296)
        flat_samples = samples[:, channel_idx, :, :].reshape(n_samples, -1).cpu().numpy()
        
        # 4. Calculate the Covariance Matrix
        # rowvar=False -> rows are observations (samples), cols are variables (pixels)
        # Result: (1296, 1296)
        cov_matrix = np.cov(flat_samples, rowvar=False)
        
        return cov_matrix, flat_samples

# ==========================================
# EXECUTE
# ==========================================
# 0 = Hall, 1 = Pedersen
cov_matrix, flat_samples = generate_vae_covariance(model, LATENT_DIM, channel_idx=1, n_samples=5000)

# ==========================================
# PLOT 1: TELECONNECTION GRID
# ==========================================
# Pick 9 interesting pixels spaced out across the center of the image
# (Avoids plotting boring black corners)
h, w = 36, 36
center_indices = np.linspace(12*36 + 12, 24*36 + 24, 9).astype(int)

fig, axs = plt.subplots(3, 3, figsize=(10, 10))
fig.suptitle("Covariance Point-Spread Functions (Teleconnections)")

for i, ax in enumerate(axs.flatten()):
    pixel_idx = center_indices[i]
    
    # Reshape the column of the covariance matrix corresponding to this pixel
    # This shows how much every other pixel varies when 'pixel_idx' varies
    teleconnection_map = cov_matrix[:, pixel_idx].reshape((h, w))
    
    # Plot
    im = ax.imshow(teleconnection_map, cmap='RdBu_r', origin='lower')
    
    # Mark the "Source" pixel
    py, px = divmod(pixel_idx, w)
    ax.scatter(px, py, color='yellow', marker='x', s=50)
    ax.set_title(f"Pixel {pixel_idx}")
    ax.axis('off')

plt.tight_layout()
plt.show()

# ==========================================
# PLOT 2: CORRELATION INTERACTIVE
# ==========================================
def plot_teleconnection(cov_matrix, pixel_idx, title=None):
    h, w = 36, 36
    
    # Convert Covariance to Correlation
    # (Cov_xy / (std_x * std_y))
    d = np.sqrt(np.diag(cov_matrix))
    
    # Avoid division by zero if some pixels (like corners) have 0 variance
    d[d == 0] = 1e-9 
    
    corr_matrix = cov_matrix / np.outer(d, d)
    
    # Extract the row for the target pixel
    row = corr_matrix[pixel_idx].reshape(h, w)
    
    plt.figure(figsize=(6, 5))
    plt.imshow(row, cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
    cbar = plt.colorbar(label='Correlation Coefficient')
    
    # Mark the seed pixel
    py, px = divmod(pixel_idx, w)
    plt.scatter(px, py, color='yellow', marker='x', s=100, label='Source')
    
    if title:
        plt.title(title)
    else:
        plt.title(f"Correlation Map for Pixel {pixel_idx}")
        
    plt.legend()
    plt.show()

# Example: Pick a pixel in the Auroral Oval (approx row 25, col 18)
pixel_idx = 25 * 36 + 18
plot_teleconnection(cov_matrix, pixel_idx, title="Auroral Oval Correlation")

#%%

import scipy.linalg

# ==========================================
# PART 1: PRINCIPAL COMPONENTS (Eigen-Auroras)
# ==========================================
def plot_eigen_modes(cov_matrix, n_modes=5):
    # 1. Eigendecomposition of the Covariance Matrix
    # eigh is for symmetric matrices (faster/more stable than eig)
    # Returns eigenvalues (w) and eigenvectors (v)
    vals, vecs = scipy.linalg.eigh(cov_matrix)
    
    # 2. Sort descending (eigh returns ascending)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    
    vals = vals / np.sum(vals) * 100
    
    # 3. Plot the top modes
    fig, axs = plt.subplots(1, n_modes, figsize=(15, 4))
    fig.suptitle(f"Top {n_modes} Learned Modes of Variation (Eigen-Auroras)")
    
    for i in range(n_modes):
        # Reshape the eigenvector (1296,) -> (36, 36)
        mode_img = vecs[:, i].reshape(36, 36)
        
        # Plot
        # We use a divergent colormap because modes have + and - regions
        limit = np.max(np.abs(mode_img))
        im = axs[i].imshow(mode_img, cmap='RdBu_r', vmin=-limit, vmax=limit, origin='lower')
        axs[i].set_title(f"Mode {i+1}\n(Var: {vals[i]:.2f})")
        axs[i].axis('off')
        
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    plt.plot(np.cumsum(vals))
    plt.xlim(-10, 100)

# Run it
plot_eigen_modes(cov_matrix, n_modes=5)

#%%
# ==========================================
# PART 2: OPTIMAL INTERPOLATION (OI)
# ==========================================
def optimal_interpolation(cov_matrix, sparse_data, sparse_mask, error_std=0.1):
    """
    Reconstructs the field using BLUE (Best Linear Unbiased Estimator).
    x_a = x_b + K (y - H x_b)
    
    Args:
        cov_matrix: (N, N) The VAE-derived covariance (P)
        sparse_data: (1, 2, 36, 36) tensor containing observations
        sparse_mask: (1, 1, 36, 36) tensor (1 where data exists)
        error_std: Assumed noise standard deviation of the measurements
    """
    # 1. Flatten Everything
    # We work with vectors of size N=1296
    h, w = 36, 36
    N = h * w
    
    # Background state (x_b): We assume 0 (Mean) since data is normalized
    x_b = np.zeros(N) 
    
    # Observations (y): Flatten the sparse data (channel 1 = Pedersen)
    y_full = sparse_data[0, 1].cpu().numpy().flatten()
    
    # Observation Operator (H): 
    # In this simple case, H is just selecting the valid indices.
    # Instead of building a massive H matrix, we just extract indices.
    mask_flat = sparse_mask[0, 0].cpu().numpy().flatten()
    valid_idx = np.where(mask_flat > 0)[0]
    
    # y vector (only valid measurements)
    y = y_full[valid_idx]
    
    # 2. Slice the Covariance Matrix (P)
    # P_Ht = P * H.T (Covariance between all points and observed points)
    P_Ht = cov_matrix[:, valid_idx]
    
    # H_P_Ht = H * P * H.T (Covariance between observed points)
    H_P_Ht = cov_matrix[np.ix_(valid_idx, valid_idx)]
    
    # 3. Observation Error Covariance (R)
    # Assume diagonal independent noise
    R = np.eye(len(valid_idx)) * (error_std**2)
    
    # 4. Calculate Kalman Gain (K)
    # K = P H^T (H P H^T + R)^-1
    # We use solve usually instead of inv for stability: K = P_Ht * inv(S)
    S = H_P_Ht + R
    K = scipy.linalg.solve(S, P_Ht.T).T # Transpose trick for solving X A = B
    
    # 5. Analysis Step (Reconstruction)
    # x_a = x_b + K (y - H x_b)
    # Since x_b is 0, this simplifies to x_a = K * y
    x_analysis = x_b + K @ (y - x_b[valid_idx])
    
    # 6. Uncertainty Estimate (Analysis Covariance diagonal)
    # P_a = (I - KH) P
    # We only need the diagonal (variance)
    # Var_a = diag(P) - diag(K * H * P)
    # This is expensive to compute fully, usually we skip or simplify.
    # Simple approx: Posterior Variance = Prior Var - Explained Var
    # explained_var = np.sum(K * P_Ht, axis=1) # roughly
    
    return x_analysis.reshape(h, w)

# ==========================================
# EXECUTE COMPARISON
# ==========================================
# Use the sparse_data_tensor from previous step
# Assume noise in measurements is low (0.05 in normalized units)
oi_reconstruction = optimal_interpolation(cov_matrix, sparse_data_tensor, sparse_mask_tensor, error_std=0.05)

# Scale back to real units
pred_OI = (oi_reconstruction + P_mu) * P_sig

# Plot Comparison
fig, axs = plt.subplots(2, 2)

ax = axs[0, 0]
plot_in = input_P.copy() # From previous context
im = ax.imshow(original_P, cmap='plasma', vmin=0, vmax=np.nanmax(original_P))
fig.colorbar(im, ax=ax)

# 1. Sparse Input
ax = axs[0, 1]
plot_in = input_P.copy() # From previous context
im = ax.imshow(plot_in, cmap='plasma', vmin=0, vmax=np.nanmax(original_P))
fig.colorbar(im, ax=ax)

# 2. VAE Optimization (Latent Recovery)
ax = axs[1, 0]
im = ax.imshow(pred_OI, cmap='plasma', vmin=0, vmax=np.nanmax(original_P))
fig.colorbar(im, ax=ax)

# 3. Covariance OI (Linear)
ax = axs[1, 1]
im = ax.imshow(original_P - pred_OI, cmap='bwr')
fig.colorbar(im, ax=ax)

plt.show()

#%%

import scipy.linalg

def reduced_rank_oi(cov_matrix, sparse_data, sparse_mask, n_modes=20, error_std=0.05):
    """
    Performs Optimal Interpolation using a Truncated (Low-Rank) Covariance Matrix.
    """
    # 1. Eigen-Decomposition
    vals, vecs = scipy.linalg.eigh(cov_matrix)
    
    # Sort descending
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    
    # 2. Truncate to top k modes
    # We only keep the signals that actually matter
    vals_k = vals[:n_modes]
    vecs_k = vecs[:, :n_modes]
    
    # 3. Reconstruct P_k (The Clean Covariance)
    # P_k = V * diag(vals) * V.T
    # This matrix is strictly positive semi-definite and noise-free
    P_k = (vecs_k * vals_k) @ vecs_k.T
    
    # 4. Standard OI with the Clean Matrix
    # (Same logic as before, just swapping P for P_k)
    h, w = 36, 36
    N = h * w
    x_b = np.zeros(N) 
    
    # Observations
    y_full = sparse_data[0, 1].cpu().numpy().flatten()
    mask_flat = sparse_mask[0, 0].cpu().numpy().flatten()
    valid_idx = np.where(mask_flat > 0)[0]
    y = y_full[valid_idx]
    
    # Slice P_k
    P_Ht = P_k[:, valid_idx]
    H_P_Ht = P_k[np.ix_(valid_idx, valid_idx)]
    
    # R Matrix (Observation Error)
    R = np.eye(len(valid_idx)) * (error_std**2)
    
    # Solve
    S = H_P_Ht + R
    # Adding a tiny jitter to diagonal for numerical stability if needed
    S += np.eye(len(valid_idx)) * 1e-6
    
    K = scipy.linalg.solve(S, P_Ht.T).T
    x_analysis = x_b + K @ (y - x_b[valid_idx])
    
    return x_analysis.reshape(h, w)

# ==========================================
# EXECUTE & COMPARE
# ==========================================

# 1. Full Rank (Noisy?)
rec_full = reduced_rank_oi(cov_matrix, sparse_data_tensor, sparse_mask_tensor, n_modes=1296)

# 2. Top 100 Modes (99% Variance)
rec_100 = reduced_rank_oi(cov_matrix, sparse_data_tensor, sparse_mask_tensor, n_modes=100)

# 3. Top 20 Modes (90% Variance - Smoothest)
rec_20 = reduced_rank_oi(cov_matrix, sparse_data_tensor, sparse_mask_tensor, n_modes=20)

# Plot
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

# Ground Truth
gt = (P_norm[id_] + P_mu) * P_sig
axs[0].imshow(gt, cmap='plasma', origin='lower')
axs[0].set_title("Ground Truth")

# Reconstructions (Scaled back)
def scale(x): return (x + P_mu) * P_sig

axs[1].imshow(scale(rec_full), cmap='plasma', origin='lower', vmin=0, vmax=np.nanmax(gt))
axs[1].set_title("Full Rank OI (1296 Modes)")

axs[2].imshow(scale(rec_100), cmap='plasma', origin='lower', vmin=0, vmax=np.nanmax(gt))
axs[2].set_title("Rank-100 OI (99% Var)")

axs[3].imshow(scale(rec_20), cmap='plasma', origin='lower', vmin=0, vmax=np.nanmax(gt))
axs[3].set_title("Rank-20 OI (90% Var)")

plt.tight_layout()
plt.show()

#%%

import scipy.linalg

def reduced_rank_oi_with_uncertainty(cov_matrix, sparse_data, sparse_mask, n_modes=20, error_std=0.05):
    """
    Performs OI and returns the Analysis Field + Uncertainty Map.
    """
    # --- 1. Eigen-Decomposition & Truncation ---
    vals, vecs = scipy.linalg.eigh(cov_matrix)
    
    # Sort descending
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    
    # Truncate
    vals_k = vals[:n_modes]
    vecs_k = vecs[:, :n_modes]
    
    # Reconstruct Clean P_k
    # P_k = V * diag(vals) * V.T
    P_k = (vecs_k * vals_k) @ vecs_k.T
    
    # Extract the Prior Variance (Diagonal of P_k)
    # This is the uncertainty BEFORE seeing any data
    prior_var = np.diag(P_k)
    
    # --- 2. Setup OI Matrices ---
    h, w = 36, 36
    N = h * w
    x_b = np.zeros(N) 
    
    # Observations
    y_full = sparse_data[0, 1].cpu().numpy().flatten()
    mask_flat = sparse_mask[0, 0].cpu().numpy().flatten()
    valid_idx = np.where(mask_flat > 0)[0]
    y = y_full[valid_idx]
    
    # Slice P_k to get Covariance between All points and Observed points
    # Shape (N, M) where M is number of observations
    P_Ht = P_k[:, valid_idx]
    
    # Covariance between Observed points and Observed points
    # Shape (M, M)
    H_P_Ht = P_k[np.ix_(valid_idx, valid_idx)]
    
    # R Matrix (Observation Error)
    R = np.eye(len(valid_idx)) * (error_std**2)
    
    # --- 3. Solve for Kalman Gain (K) ---
    S = H_P_Ht + R
    # Tiny jitter for stability
    S += np.eye(len(valid_idx)) * 1e-8
    
    # K = P H^T (S)^-1
    # Shape (N, M)
    K = scipy.linalg.solve(S, P_Ht.T).T
    
    # --- 4. Analysis Step (The Field) ---
    x_analysis = x_b + K @ (y - x_b[valid_idx])
    
    # --- 5. Uncertainty Step (The Variance) ---
    # Formula: Var_a = Var_b - diag(K * H * P)
    # Note: H*P is exactly P_Ht.T (Cov between obs and grid)
    # We calculate the "Explained Variance" term efficiently:
    # explained_var = sum(K * P_Ht, axis=1)
    explained_var = np.sum(K * P_Ht, axis=1)
    
    # Posterior Variance
    analysis_var = prior_var - explained_var
    
    # Convert to Sigma (Standard Deviation) for plotting
    # Clip to 0 to avoid tiny negative numbers from float errors
    analysis_sigma = np.sqrt(np.maximum(analysis_var, 0))
    
    return x_analysis.reshape(h, w), analysis_sigma.reshape(h, w)

# ==========================================
# EXECUTE & PLOT
# ==========================================

# Run OI with uncertainty
rec_field, rec_sigma = reduced_rank_oi_with_uncertainty(
    cov_matrix, 
    sparse_data_tensor, 
    sparse_mask_tensor, 
    n_modes=20,       # Using top 50 modes
    error_std=0.05    # Assume 5% measurement error
)

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# 1. The Reconstruction
# Scale back to physical units
rec_phys = (rec_field + P_mu) * P_sig
# Mask out the background for cleaner look? Optional.
im1 = axs[0].imshow(rec_phys, cmap='plasma', origin='lower')
axs[0].set_title("Reconstructed Conductance")
plt.colorbar(im1, ax=axs[0], label="Conductance (S)")

# 2. The Uncertainty Map (Sigma)
# This shows WHERE the model is unsure.
# It should be low near data points, and high in the gaps.
im2 = axs[1].imshow(rec_sigma, cmap='viridis', origin='lower')
axs[1].set_title("Model Uncertainty ($1\sigma$)")
plt.colorbar(im2, ax=axs[1], label="Normalized Units")

# 3. Overlay Data on Uncertainty
# To verify that uncertainty drops near measurements
axs[2].imshow(rec_sigma, cmap='viridis', origin='lower')
# Plot measurements as red dots
y_idxs, x_idxs = np.where(sparse_mask_tensor[0,0].cpu().numpy() > 0)
axs[2].scatter(x_idxs, y_idxs, c='r', s=2, alpha=0.5)
axs[2].set_title("Uncertainty + Data Locations")

plt.tight_layout()
plt.show()


#%%


def generate_log_statistics(model, latent_dim, channel_idx=1, n_samples=5000, epsilon=1.0):
    """
    Generates Mean and Covariance of the LOG-TRANSFORMED conductance.
    x_log = ln(x_physical + epsilon)
    """
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        # 1. Sample Latent Space
        z = torch.randn(n_samples, latent_dim).to(device)
        samples_norm = model.decode(z)[:, channel_idx, :, :] # (N, 36, 36)
        
        # 2. De-Normalize to Physical Units (Siemens)
        # Using the global P_mu, P_sig you calculated earlier
        samples_phys = (samples_norm.cpu().numpy() * P_sig) + P_mu
        
        # Clip to 0 just in case the VAE generated a tiny negative value
        samples_phys = np.maximum(samples_phys, 0)
        
        # 3. Log Transform
        # We add epsilon (e.g., 1.0 S) to handle regions with 0 conductance
        # and to keep the math stable.
        samples_log = np.log(samples_phys + epsilon)
        
        # 4. Flatten
        samples_flat = samples_log.reshape(n_samples, -1)
        
        # 5. Calculate Log-Mean and Log-Covariance
        mean_log = np.mean(samples_flat, axis=0)
        cov_log = np.cov(samples_flat, rowvar=False)
        
        return mean_log, cov_log

# Execute
# Note: Ensure P_mu and P_sig are available and correct!
log_mean_vec, log_cov_matrix = generate_log_statistics(model, LATENT_DIM, channel_idx=1)

#%%

def log_space_oi(log_mean, log_cov, sparse_data_norm, sparse_mask, error_std_phys=0.5, epsilon=1.0, n_modes=50):
    """
    Performs OI in Log-Space to guarantee positive output.
    """
    # --- 1. Transform Observations to Log-Physical Units ---
    # Get normalized data
    y_norm = sparse_data_norm[0, 1].cpu().numpy().flatten()
    mask_flat = sparse_mask[0, 0].cpu().numpy().flatten()
    valid_idx = np.where(mask_flat > 0)[0]
    
    # De-normalize to Physical
    y_phys = (y_norm[valid_idx] * P_sig) + P_mu
    y_phys = np.maximum(y_phys, 0) # Safety clip
    
    # Log Transform Observations
    y_log = np.log(y_phys + epsilon)
    
    # --- 2. Reduced Rank OI Setup (Same as before, but using Log stats) ---
    # Decompose the Log-Covariance
    vals, vecs = scipy.linalg.eigh(log_cov)
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx][:n_modes], vecs[:, idx][:, :n_modes]
    
    # Reconstruct Clean Log-Covariance
    P_k = (vecs * vals) @ vecs.T
    
    # Prior (Background) is now the Log-Mean (not zero!)
    x_b = log_mean
    
    # Slice P_k
    P_Ht = P_k[:, valid_idx]
    H_P_Ht = P_k[np.ix_(valid_idx, valid_idx)]
    
    # --- 3. Observation Error in Log-Space ---
    # Error of 0.5 S in physical space != 0.5 in log space.
    # Propagation of uncertainty: sigma_log approx sigma_phys / (mu_phys + epsilon)
    # For simplicity, we can assume a fixed log-error, e.g., 0.1 (approx 10% relative error)
    # or calculate it per observation:
    sigma_log = error_std_phys / (y_phys + epsilon)
    R = np.diag(sigma_log**2)
    
    # --- 4. Solve ---
    S = H_P_Ht + R + np.eye(len(valid_idx)) * 1e-6
    K = scipy.linalg.solve(S, P_Ht.T).T
    
    # Analysis in Log-Space
    # x_a = x_b + K (y - H x_b)
    x_analysis_log = x_b + K @ (y_log - x_b[valid_idx])
    
    # --- 5. Inverse Transform (Exp) ---
    # x_phys = exp(x_log) - epsilon
    x_analysis_phys = np.exp(x_analysis_log) - epsilon
    
    # Final safety clip (removes -1e-8 type floating point errors)
    x_analysis_phys = np.maximum(x_analysis_phys, 0)
    
    return x_analysis_phys.reshape(36, 36)

# ==========================================
# EXECUTE
# ==========================================
# Run the Log-Space OI
# epsilon=1.0 means we are modeling ln(Conductance + 1)
# error_std_phys=0.5 means we assume measurements have +/- 0.5 S noise
rec_positive = log_space_oi(
    log_mean_vec, 
    log_cov_matrix, 
    sparse_data_tensor, 
    sparse_mask_tensor, 
    error_std_phys=0.5, 
    epsilon=1.0,
    n_modes=20
)

# Plot
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Ground Truth (Physical)
gt_phys = (P_norm[id_] * P_sig) + P_mu
im1 = axs[0].imshow(gt_phys, cmap='plasma', origin='lower', vmin=0)
axs[0].set_title("Ground Truth (Siemens)")
plt.colorbar(im1, ax=axs[0])

# Log-OI Reconstruction (Guaranteed Positive)
im2 = axs[1].imshow(rec_positive, cmap='plasma', origin='lower', vmin=0, vmax=np.nanmax(gt_phys))
axs[1].set_title("Log-Space OI Reconstruction")
plt.colorbar(im2, ax=axs[1])

plt.show()

# Verification
print(f"Minimum Value in Reconstruction: {np.min(rec_positive):.4f}")

#%%

def log_space_oi_with_uncertainty(log_mean, log_cov, sparse_data_norm, sparse_mask, error_std_phys=0.5, epsilon=1.0, n_modes=50):
    """
    Performs Log-Space OI and returns Physical Map + Physical Uncertainty.
    """
    # --- 1. PREPARE DATA ---
    h, w = 36, 36
    # Get normalized data
    y_norm = sparse_data_norm[0, 1].cpu().numpy().flatten()
    mask_flat = sparse_mask[0, 0].cpu().numpy().flatten()
    valid_idx = np.where(mask_flat > 0)[0]
    
    # De-normalize to Physical Units (Siemens)
    y_phys = (y_norm[valid_idx] * P_sig) + P_mu
    y_phys = np.maximum(y_phys, 0) # Clip
    
    # Log Transform Observations
    y_log = np.log(y_phys + epsilon)
    
    # --- 2. TRUNCATED LOG-COVARIANCE ---
    vals, vecs = scipy.linalg.eigh(log_cov)
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx][:n_modes], vecs[:, idx][:, :n_modes]
    
    # Reconstruct Clean P_k
    P_k = (vecs * vals) @ vecs.T
    
    # Get Prior Variance (Diagonal of P_k in log space)
    prior_var_log = np.diag(P_k)
    
    # --- 3. SOLVE OI ---
    x_b = log_mean
    P_Ht = P_k[:, valid_idx]
    H_P_Ht = P_k[np.ix_(valid_idx, valid_idx)]
    
    # R Matrix: Convert physical error (e.g. 0.5 S) to log-error
    # sigma_log approx sigma_phys / (value + epsilon)
    sigma_log_obs = error_std_phys / (y_phys + epsilon)
    R = np.diag(sigma_log_obs**2)
    
    # Kalman Gain
    S = H_P_Ht + R + np.eye(len(valid_idx)) * 1e-6
    K = scipy.linalg.solve(S, P_Ht.T).T
    
    # Analysis Mean (Log Space)
    x_analysis_log = x_b + K @ (y_log - x_b[valid_idx])
    
    # Analysis Variance (Log Space)
    # var_a = var_b - explained_var
    explained_var = np.sum(K * P_Ht, axis=1)
    analysis_var_log = prior_var_log - explained_var
    analysis_var_log = np.maximum(analysis_var_log, 0) # Safety clip
    
    # --- 4. CONVERT TO PHYSICAL UNITS ---
    
    # A. Physical Mean (Median of Log-Normal)
    x_phys = np.exp(x_analysis_log) - epsilon
    x_phys = np.maximum(x_phys, 0)
    
    # B. Physical Uncertainty (Standard Deviation)
    # Formula: sigma_phys = (Mean_phys + eps) * sqrt(exp(sigma_log^2) - 1)
    # This captures the multiplicative nature: higher signal = higher absolute error
    sigma_factor = np.sqrt(np.exp(analysis_var_log) - 1)
    sigma_phys = (x_phys + epsilon) * sigma_factor
    
    return x_phys.reshape(h, w), sigma_phys.reshape(h, w)

# ==========================================
# EXECUTE & PLOT
# ==========================================

rec_phys, unc_phys = log_space_oi_with_uncertainty(
    log_mean_vec, 
    log_cov_matrix, 
    sparse_data_tensor, 
    sparse_mask_tensor, 
    error_std_phys=0.5, 
    epsilon=1.0, 
    n_modes=20
)

# Visualization
fig, axs = plt.subplots(1, 4, figsize=(15, 5))

# 1. Reconstruction
im1 = axs[0].imshow(rec_phys, cmap='plasma', origin='lower', vmin=0)
axs[0].set_title("Reconstruction (Positivity Guaranteed)")
plt.colorbar(im1, ax=axs[0], label="Siemens")

# 2. Uncertainty
im2 = axs[1].imshow(unc_phys, cmap='viridis', origin='lower', vmin=0)
axs[1].set_title("Uncertainty ($1\sigma$)")
plt.colorbar(im2, ax=axs[1], label="Siemens")

# 3. SNR (Signal to Noise Ratio)
# This is a great diagnostic. If > 1, the signal is trustworthy.
snr = rec_phys / (unc_phys + 1e-6)
im3 = axs[2].imshow(snr, cmap='inferno', origin='lower', vmin=0, vmax=10)
axs[2].set_title("Signal-to-Noise Ratio (Mean / Uncertainty)")
plt.colorbar(im3, ax=axs[2])

# 3. SNR (Signal to Noise Ratio)
# This is a great diagnostic. If > 1, the signal is trustworthy.
snr = rec_phys / (unc_phys + 1e-6)
im3 = axs[3].imshow(unc_phys, cmap='inferno', origin='lower', vmin=0, vmax=2)
plt.colorbar(im3, ax=axs[3])


plt.tight_layout()
plt.show()








