import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. PREPARE DATA (Normalize & Mask)
# ==========================================
# Assuming X is your (N, 36, 36) array with NaNs

# 1. Normalize to [0, 1] (Crucial for VAE Sigmoid output)

X_trunc = copy.deepcopy(X)
data_min = np.nanmin(X_trunc)
data_max = np.nanmax(X_trunc)
X_norm = (X_trunc[:1000] - data_min) / (data_max - data_min)

# 2. Convert to Tensor
tensor_x = torch.as_tensor(X_norm).float()

# 3. Create Mask (1=Valid, 0=NaN)
mask = ~torch.isnan(tensor_x)

# 4. Fill NaNs with 0 (safe for network input)
tensor_x[~mask] = 0.0

# 5. Create Loader
dataset = TensorDataset(tensor_x.unsqueeze(1), mask.float().unsqueeze(1))
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# ==========================================
# 2. DEFINE VAE MODEL
# ==========================================
class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(VAE, self).__init__()
        
        # --- ENCODER ---
        # Input: (1, 36, 36)
        self.enc1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1) # -> (32, 18, 18)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # -> (64, 9, 9)
        self.enc3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # -> (128, 9, 9)
        
        self.flatten_dim = 128 * 9 * 9
        
        # Latent vectors
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # --- DECODER ---
        self.fc_dec = nn.Linear(latent_dim, self.flatten_dim)
        
        self.dec1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1) # -> (64, 9, 9)
        self.dec2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # -> (32, 18, 18)
        self.dec3 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)   # -> (1, 36, 36)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x_flat = x.view(x.size(0), -1)
        
        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x = self.fc_dec(z)
        x = x.view(x.size(0), 128, 9, 9)
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        # Sigmoid forces output to [0, 1]
        recon = torch.sigmoid(self.dec3(x)) 
        
        return recon, mu, logvar

model = VAE(latent_dim=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ==========================================
# 3. LOSS FUNCTION (Masked)
# ==========================================
def loss_function(recon_x, x, mask, mu, logvar):
    # 1. Masked Reconstruction Loss (MSE)
    # We only punish error where the mask is 1
    # 'sum' or 'mean'? 'sum' often works better for VAEs to balance KLD
    recon_loss = F.mse_loss(recon_x * mask, x * mask, reduction='sum')
    
    # Normalize by number of valid pixels so it's not dependent on gap size
    valid_pixels = mask.sum() + 1e-8
    recon_loss = recon_loss / valid_pixels * (36*36) # Scale to full image size equivalent

    # 2. KLD Loss (Regularization) - Forces latent space to be Gaussian
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Scale KLD usually by batch size, handled by reduction='sum' above for recon?
    # Standard VAE: Recon + KLD.
    # Often KLD is too strong, so we use a beta factor (e.g. 0.1 * KLD)
    
    return recon_loss + 0.1 * kld_loss

# ==========================================
# 4. TRAINING LOOP
# ==========================================
n_epochs = 2000

print(f"Training VAE on {device}...")
model.train()

for epoch in range(n_epochs):
    train_loss = 0
    for batch_img, batch_mask in dataloader:
        batch_img = batch_img.to(device)
        batch_mask = batch_mask.to(device)
        
        # Forward
        recon_batch, mu, logvar = model(batch_img)
        
        # Loss
        loss = loss_function(recon_batch, batch_img, batch_mask, mu, logvar)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {train_loss / len(dataloader.dataset):.4f}")

#%%

# ==========================================
# 5. RECONSTRUCT & VISUALIZE
# ==========================================
id_ = 800
model.eval()
with torch.no_grad():
    # Grab first sample
    sample_img = dataset[id_][0].unsqueeze(0).to(device) # (1, 1, 36, 36)
    sample_mask = dataset[id_][1].unsqueeze(0).to(device)
    
    # Reconstruct
    recon, _, _ = model(sample_img)
    
    # Convert back to numpy and un-normalize
    orig = sample_img.cpu().squeeze().numpy()
    rec = recon.cpu().squeeze().numpy()
    msk = sample_mask.cpu().squeeze().numpy()
    
    # Scale back to original units
    orig_scaled = (orig * (data_max - data_min)) + data_min
    rec_scaled = (rec * (data_max - data_min)) + data_min
    
    # For visualization, set NaN in original where mask is 0
    orig_scaled[msk == 0] = np.nan

    # Plot
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 3, 1)
    plt.title("Input (with Gaps)")
    plt.imshow(orig_scaled, cmap='plasma', origin='lower')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.title("VAE Reconstruction")
    plt.imshow(rec_scaled, cmap='plasma', origin='lower')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(rec_scaled, cmap='gray', origin='lower', alpha=0.5)
    plt.imshow(orig_scaled, cmap='plasma', origin='lower', alpha=0.8) # Valid pixels on top
    
    plt.tight_layout()
    plt.show()

#%%

img_true = X_trunc[-1000]

img_sub = img_true.flatten()
np.random.seed(1)
img_sub[np.random.randint(0, img_sub.size-1, 100)] = np.nan
img_sub = torch.as_tensor(img_sub.reshape(img_true.shape)).float()

nan_mask = ~torch.isnan(img_sub)
img_filled = img_sub.clone()
img_filled[~nan_mask] = 0.0
nan_mask = nan_mask.float()

model.eval()
with torch.no_grad():
    sample_img = img_filled.unsqueeze(0).unsqueeze(0).to(device) # (1, 1, 36, 36)
    sample_mask = nan_mask.unsqueeze(0).unsqueeze(0).to(device) # (1, 1, 36, 36)
    recon, _, _ = model(sample_img)

fig, axs = plt.subplots(1, 3)
axs[0].imshow(img_true, cmap='plasma', origin='lower')
axs[1].imshow(img_filled, cmap='plasma', origin='lower')
axs[2].imshow(recon.squeeze(), cmap='plasma', origin='lower')






























