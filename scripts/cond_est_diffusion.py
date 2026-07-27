import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. PREPARE YOUR DATA (X is given)
# ==========================================
# Assumes X is defined in your variable explorer
# If testing, uncomment the next line:
# X = np.random.randn(100, 36, 36).astype(np.float32); X[np.random.rand(*X.shape)>0.7] = np.nan

scale_factor = np.nanmax(X) 

print(f"Normalizing data by dividing by {scale_factor}")
X_normalized = X / scale_factor

tensor_x = torch.as_tensor(X_normalized[:1000]).float()
mask = ~torch.isnan(tensor_x) # True = Valid, False = NaN

# Fill NaNs with 0.0 for the network input (we mask the loss later)
# We clone to avoid modifying your original X in place if that matters
tensor_x_filled = tensor_x.clone()
tensor_x_filled[~mask] = 0.0

# Create DataLoader: Yields (Image, Mask)
dataset = TensorDataset(tensor_x_filled.unsqueeze(1), mask.float().unsqueeze(1))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ==========================================
# 2. DEFINE THE MODEL (FIXED DIMENSIONS)
# ==========================================
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Time embedding: Output must match enc2 channels (64)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 32), 
            nn.ReLU(), 
            nn.Linear(32, 64) # <--- FIXED: Was 32, now 64
        )
        
        # Encoder
        self.enc1 = nn.Conv2d(1, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, padding=1)
        
        # Decoder
        self.dec1 = nn.ConvTranspose2d(64, 32, 3, padding=1)
        self.dec2 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x, t):
        # Embed time
        t_emb = self.time_mlp(t).view(-1, 64, 1, 1) # <--- FIXED: Reshape to 64
        
        # Encode
        x1 = F.relu(self.enc1(x))
        x2 = F.relu(self.enc2(x1))
        
        # Inject time
        x2 = x2 + t_emb
        
        # Decode
        x3 = F.relu(self.dec1(x2))
        return self.dec2(x3)

model = SimpleUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ==========================================
# 3. TRAIN
# ==========================================
n_epochs = 1000
n_timesteps = 500

print(f"Starting training on {device}...")
model.train()

for epoch in range(n_epochs):
    epoch_loss = 0
    for batch_img, batch_mask in dataloader:
        batch_img = batch_img.to(device)
        batch_mask = batch_mask.to(device)
        
        # Random Timesteps
        t = torch.randint(0, n_timesteps, (batch_img.size(0), 1)).float().to(device)
        noise = torch.randn_like(batch_img).to(device)
        
        # Add Noise
        alpha = 1 - (t / n_timesteps).view(-1, 1, 1, 1)
        noisy_input = batch_img * alpha + noise * (1 - alpha)
        
        # Predict Noise
        predicted_noise = model(noisy_input, t)
        
        # MASKED LOSS: Only learn from valid pixels
        loss = F.mse_loss(predicted_noise * batch_mask, noise * batch_mask)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {epoch_loss/len(dataloader):.5f}")

# ==========================================
# 4. RECONSTRUCT (Inpainting Function)
# ==========================================
@torch.no_grad()
def reconstruct(model, partial_img_tensor, mask_tensor, n_steps=50):
    model.eval()
    # Prepare inputs
    x = torch.randn(1, 1, 36, 36).to(device)
    img = partial_img_tensor.unsqueeze(0).to(device)
    msk = mask_tensor.unsqueeze(0).to(device)

    for i in reversed(range(n_steps)):
        t = torch.full((1, 1), i, dtype=torch.float32).to(device)
        
        # Denoise step
        pred_noise = model(x, t)
        alpha = 1 - (i / n_steps)
        x = x - pred_noise * 0.1 
        
        # Inpainting step
        if i > 0:
            noise = torch.randn_like(x) * (1 - alpha)
            noisy_ground_truth = img * alpha + noise
            x = noisy_ground_truth * msk + x * (1 - msk)
            
    return x.squeeze().cpu().numpy()

# Test on first sample
print("Reconstructing first sample...")
# Make sure to pass the Tensors from the dataset, not the numpy array
rec_image = reconstruct(model, dataset[0][0], dataset[0][1])

plt.figure(figsize=(8,4))
plt.subplot(1,2,1); plt.title("Original (0=NaN)"); plt.imshow(dataset[0][0].squeeze())
plt.subplot(1,2,2); plt.title("Reconstructed"); plt.imshow(rec_image)
plt.show()

#%%

# Test on first sample
print("Reconstructing first sample...")
# Make sure to pass the Tensors from the dataset, not the numpy array
q_ = X[-1000].flatten()
q_[np.random.randint(0, q_.size-1, 3000)] = np.nan
q_ = q_.reshape(X[0].shape)
q = torch.as_tensor(q_).float()
qf = ~torch.isnan(q) # True = Valid, False = NaN
q_filled = q.clone()
q_filled[~qf] = 0.0
qf = qf.float()

rec_image = reconstruct(model, q_filled, qf)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1); plt.title("Original (0=NaN)"); plt.imshow(q_filled.squeeze())
plt.subplot(1,2,2); plt.title("Reconstructed"); plt.imshow(rec_image)
plt.show()
