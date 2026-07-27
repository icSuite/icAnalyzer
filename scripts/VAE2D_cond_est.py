#%% Import
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import glob
from icreader import ConductanceImage # Commented out for portability, un-comment in your env
from tqdm import tqdm

from os.path import join as pjoin
import pandas as pd
import matplotlib.pyplot as plt

#%% Fun

class VAE2D(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE2D, self).__init__()
        self.latent_dim = latent_dim

        # --- ENCODER ---
        # Input shape: (Batch, Channels, 36, 36)
        
        # 1. 36 -> 18
        self.enc_conv1 = nn.Conv2d(CHANNELS, 16, kernel_size=3, stride=2, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(16)
        
        # 2. 18 -> 9
        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(32)
        
        # 3. 9 -> 5
        self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(64)
        
        # Flatten: 64 channels * 5 height * 5 width
        self.flatten_dim = 64 * 5 * 5 
        
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # --- DECODER ---
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        
        # 1. Input (64, 5, 5) -> Target (32, 9, 9)
        # (5-1)*2 - 2*1 + 3 + op = 9  =>  8 - 2 + 3 + op = 9  => op=0
        self.dec_conv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.dec_bn1 = nn.BatchNorm2d(32)
        
        # 2. Input (32, 9, 9) -> Target (16, 18, 18)
        # (9-1)*2 - 2*1 + 3 + op = 18 => 16 - 2 + 3 + op = 18 => op=1
        self.dec_conv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn2 = nn.BatchNorm2d(16)
        
        # 3. Input (16, 18, 18) -> Target (CHANNELS, 36, 36)
        # (18-1)*2 - 2*1 + 3 + op = 36 => 34 - 2 + 3 + op = 36 => op=1
        self.dec_conv3 = nn.ConvTranspose2d(16, CHANNELS, kernel_size=3, stride=2, padding=1, output_padding=1)

    def encode(self, x):
        x = F.leaky_relu(self.enc_bn1(self.enc_conv1(x)))
        x = F.leaky_relu(self.enc_bn2(self.enc_conv2(x)))
        x = F.leaky_relu(self.enc_bn3(self.enc_conv3(x)))
        
        x_flat = x.view(x.size(0), -1)
        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

#    def decode(self, z):
#        x = self.fc_decode(z)
#        # Reshape to 2D tensor (Batch, Channels, H, W)
#        x = x.view(x.size(0), 64, 5, 5) 
#        
#        x = F.leaky_relu(self.dec_bn1(self.dec_conv1(x)))
#        x = F.leaky_relu(self.dec_bn2(self.dec_conv2(x)))
#        x = self.dec_conv3(x) 
#        
#        return x
    
    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(x.size(0), 64, 5, 5) 
        x = F.leaky_relu(self.dec_bn1(self.dec_conv1(x)))
        x = F.leaky_relu(self.dec_bn2(self.dec_conv2(x)))
        x = self.dec_conv3(x) 
        
        # Optional: Force the output to be 2 channels if you only want 
        # to reconstruct H and P, or keep 3 if you want it to reconstruct the mask.
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

#def loss_function(recon, x, mu, logvar, beta=1):
#    recon_loss = F.mse_loss(recon, x, reduction='sum')
#    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#    return recon_loss + beta * kld_loss

def loss_function(recon, x, mu, logvar, beta=1):
    # x[:, 2:3] is the mask channel we just added
    mask = x[:, 2:3, :, :] 
    
    # MSE only on the Hall and Pedersen channels (0 and 1)
    # where the mask is 1
    recon_loss = F.mse_loss(recon[:, :2] * mask, x[:, :2] * mask, reduction='sum')
    
    # Standard KLD
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kld_loss

# --- Main Execution Block ---
# Assuming you have run the cells above and 'X_all' exists in memory as numpy array
# X_all shape is currently: (N, 10, 5, 36, 36) -> (N, Time, Channel, H, W)

#%% Hyper paramaters

CHANNELS = 3           # Hall, Hall_Unc, Pedersen, Pedersen_Unc, Mask
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

#H = H[:, :, 30:] = np.nan
#P = P[:, :, 30:] = np.nan

h, w = 36, 36

f = np.sum((np.isnan(H) | np.isnan(P)), axis=(1,2)) / (h*w)
f = f < .33

H, P = H[f], P[f]

y, x = np.ogrid[:h, :w]
mask_circular = (x - w/2)**2 + (y - h/2)**2 > (w/2)**2
H[:, mask_circular] = np.nan 
P[:, mask_circular] = np.nan 

#%%

H_mu, H_sig = np.nanmean(H), np.nanstd(H)
P_mu, P_sig = np.nanmean(P), np.nanstd(P)

H_norm = (H-H_mu)/H_sig
P_norm = (P-P_mu)/P_sig

H_trunc = H_norm[:10000]
P_trunc = P_norm[:10000]
mask_np = (~np.isnan(H_trunc)).astype(np.float32) 

# 1. Stack Hall and Pedersen into Channel dimension
# H_trunc shape: (N, 36, 36)
# P_trunc shape: (N, 36, 36)
# Result X shape: (N, 2, 36, 36)
X_np = np.stack([np.nan_to_num(H_trunc, nan=0.0), 
                 np.nan_to_num(P_trunc, nan=0.0),
                 mask_np], 
                axis=1)

# 2. Handle NaNs (Crucial: VAE cannot handle NaNs)
# Since you already normalized (mean=0), filling with 0 is statistically safe-ish
X_np = np.nan_to_num(X_np, nan=0.0)

# 3. Create Tensor
tensor_x = torch.from_numpy(X_np).float()

#%%

# Create DataLoader
dataloader = DataLoader(tensor_x, batch_size=BATCH_SIZE, shuffle=True)

# Initialize Model with Correct Channels
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE2D(latent_dim=LATENT_DIM).to(device) # Make sure class uses global CHANNELS=2
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Model initialized on {device} with {CHANNELS} channels")

# Training Loop
num_epochs = 100 

model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, data in enumerate(dataloader):
        # Input data is already (Batch, Channel, Height, Width) -> (8, 2, 36, 36)
        data = data.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        recon_batch, mu, logvar = model(data)
        
        # Loss
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    # Calculate average loss over the dataset
    avg_loss = total_loss / len(dataloader.dataset)
    print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')

#%%

recon, mu, logvar = model(tensor_x)

with torch.no_grad():
    fig, axs = plt.subplots(2, 5)
    for i in range(5):
        axs[0, i].imshow(tensor_x[i*50, 0], cmap='plasma')
        axs[1, i].imshow(recon[i*50, 0], cmap='plasma')

#%%

id_ = 3010
np.random.seed(1)

H_, P_ = H_norm[id_].flatten(), P_norm[id_].flatten()
idx = np.unique(np.random.randint(0, 36*36-1, 2000))
H_[idx], P_[idx] = np.nan, np.nan
mask = (~(np.isnan(H_) | np.isnan(P_))).astype(np.float32).reshape((36, 36))
H_ = np.nan_to_num(H_, nan=0.0).reshape((36, 36))
P_ = np.nan_to_num(P_, nan=0.0).reshape((36, 36))

X_norm = np.stack([H_, P_, mask], axis=0)

tX_norm = torch.from_numpy(X_norm).float()
recon, mu, logvar = model(tX_norm.unsqueeze(0))

x_ = (P_norm[id_] + P_mu)*P_sig
x_nan = (tX_norm[1] + P_mu)*P_sig
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

def reconstruct_from_sparse(model, sparse_data, sparse_mask, iterations=100):
    model.eval()
    # 1. Start with a random latent vector z
    z = torch.randn(1, LATENT_DIM, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=0.1)
    
    for i in range(iterations):
        optimizer.zero_grad()
        
        # 2. Decode current z into a full map
        recon = model.decode(z) # (1, 2, 36, 36)
        
        # 3. Calculate loss ONLY at the sparse measurement locations
        # Compare recon (channels 0 & 1) to sparse_data
        loss = F.mse_loss(recon[:, :2] * sparse_mask, sparse_data * sparse_mask)
        
        # 4. Backprop to nudge z
        loss.backward()
        optimizer.step()
        
    return model.decode(z).detach().cpu().numpy()

id_ = 3010
np.random.seed(1)

H_, P_ = H_norm[id_].flatten(), P_norm[id_].flatten()
idx = np.unique(np.random.randint(0, 36*36-1, 2000))
H_[idx], P_[idx] = np.nan, np.nan
mask = (~(np.isnan(H_) | np.isnan(P_))).astype(np.float32).reshape((36, 36))
H_ = np.nan_to_num(H_, nan=0.0).reshape((36, 36))
P_ = np.nan_to_num(P_, nan=0.0).reshape((36, 36))

sparse_data = torch.from_numpy(np.stack([H_, P_], axis=0)).float().unsqueeze(0)
sparse_mask = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)

# Usage:
# sparse_data: (1, 2, 36, 36) tensor, 0s everywhere except measurements
# sparse_mask: (1, 1, 36, 36) tensor, 1 at measurement locations, 0 elsewhere
full_map = reconstruct_from_sparse(model, sparse_data, sparse_mask)

x_ = (P_norm[id_] + P_mu)*P_sig
x_nan = (sparse_data[0, 1]+P_mu)*P_sig
pred = (full_map[0, 1]+P_mu)*P_sig

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


def generate_vae_covariance(model, latent_dim, n_samples=2000):
    model.eval()
    with torch.no_grad():
        # 1. Sample from the Latent Prior (Standard Normal)
        z = torch.randn(n_samples, latent_dim).to(device)
        
        # 2. Decode into Images
        # Output shape: (n_samples, CHANNELS, 36, 36)
        samples = model.decode(z)
        
        # 3. Focus on a specific channel (e.g., Hall Conductance)
        # Flatten: (n_samples, 1296)
        hall_samples = samples[:, 0, :, :].reshape(n_samples, -1).cpu().numpy()
        
        # 4. Calculate the Covariance Matrix
        # Result: (1296, 1296)
        cov_matrix = np.cov(hall_samples, rowvar=False)
        
        return cov_matrix, hall_samples

# Execute
cov_matrix, hall_samples = generate_vae_covariance(model, LATENT_DIM)

plt.imshow(cov_matrix[:,0].reshape((36,36)))

#%%

def plot_teleconnection(cov_matrix, pixel_idx, title="Correlation Map"):
    # Convert Covariance to Correlation for better visualization
    d = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(d, d)
    
    # Extract the row for the target pixel
    row = corr_matrix[pixel_idx].reshape(36, 36)
    
    plt.imshow(row, cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
    plt.colorbar(label='Correlation Coefficient')
    # Mark the seed pixel
    plt.scatter(pixel_idx % 36, pixel_idx // 36, color='yellow', marker='x')
    plt.title(title)
    plt.show()

# Example: Pick a pixel in the auroral zone
plot_teleconnection(cov_matrix, pixel_idx=600)
