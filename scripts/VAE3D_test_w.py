#%% Import
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import glob
from icreader import ConductanceImage # Commented out for portability, un-comment in your env
from tqdm import tqdm

#%% Fun

class VAE3D(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE3D, self).__init__()
        self.latent_dim = latent_dim

        # --- ENCODER ---
        # Input shape: (Batch, 5, 10, 36, 36)
        
        # 1. (36x36) -> (18x18), Time 10 -> 5
        self.enc_conv1 = nn.Conv3d(in_channels=CHANNELS, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.enc_bn1 = nn.BatchNorm3d(16)
        
        # 2. (18x18) -> (9x9), Time 5 -> 3
        self.enc_conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm3d(32)
        
        # 3. (9x9) -> (5x5), Time 3 -> 2
        self.enc_conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.enc_bn3 = nn.BatchNorm3d(64)
        
        # Flatten for linear layers
        # Dimensions at this depth: 64 channels, 2 time steps, 5 height, 5 width
        self.flatten_dim = 64 * 2 * 5 * 5 
        
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # --- DECODER ---
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        
        # 1. Unflatten to (64, 2, 5, 5)
        self.dec_conv1 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=(0,1,1))
        self.dec_bn1 = nn.BatchNorm3d(32)
        # Output spatial: 5 -> 9 (roughly)
        
        self.dec_conv2 = nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=(1,1,1))
        self.dec_bn2 = nn.BatchNorm3d(16)
        # Output spatial: 9 -> 18
        
        self.dec_conv3 = nn.ConvTranspose3d(in_channels=16, out_channels=CHANNELS, kernel_size=3, stride=2, padding=1, output_padding=(0,1,1))
        # Output spatial: 18 -> 36. 
        # Note: Tuning output_padding is tricky. If 35, we interpolate. If 36, perfect.
        # With kernel 3, stride 2, pad 1, in 18: (18-1)*2 -2 + 3 + op = 35 + op. So op=1 makes it 36.

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

    def decode(self, z):
        x = self.fc_decode(z)
        # Reshape to correct tensor volume
        x = x.view(x.size(0), 64, 2, 5, 5) 
        
        x = F.leaky_relu(self.dec_bn1(self.dec_conv1(x)))
        x = F.leaky_relu(self.dec_bn2(self.dec_conv2(x)))
        x = self.dec_conv3(x) 
        
        # Final safety resize to ensure exactly (10, 36, 36)
        if x.shape[2] != SEQUENCE_LENGTH or x.shape[3] != IMG_HEIGHT:
            x = F.interpolate(x, size=(SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH), mode='trilinear', align_corners=False)
            
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def loss_function(recon, x, mu, logvar, beta=1):
    recon_loss = F.mse_loss(recon, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld_loss

# --- Main Execution Block ---
# Assuming you have run the cells above and 'X_all' exists in memory as numpy array
# X_all shape is currently: (N, 10, 5, 36, 36) -> (N, Time, Channel, H, W)

#%% Hyper paramaters

SEQUENCE_LENGTH = 10   
CHANNELS = 1           # Hall, Hall_Unc, Pedersen, Pedersen_Unc, Mask
IMG_HEIGHT = 36        
IMG_WIDTH = 36         
LATENT_DIM = 36        
BATCH_SIZE = 8
LEARNING_RATE = 1e-4

#%% Paths & Fetch Orbits

p_in = '/home/bing/Dropbox/work/data/conductance/'
o = [int(o[-7:-3]) for o in sorted(glob.glob(p_in + '*.nc'))]

#%% Load raw data (Same as before)

H, P, dH, dP, w = [], [], [], [], []
for orbit in tqdm(o, total=len(o)):
    filename = p_in + f'or_{str(orbit).zfill(4)}.nc'
    cI = ConductanceImage(filename)
    H.append(cI.H)
    P.append(cI.P)
    dH.append(cI.dH)
    dP.append(cI.dP)
    w.append(cI.w)

#%% Load labels

q = np.load('/home/bing/Dropbox/work/temp_storage/seq_classification_results.npz')
orbits_idx = q['orbits']
indices = q['indices']
labels = q['labels']

#%% Prepare sequences
def prepare_seq(w_, idx):
    w = w_[idx:idx+10][:, None, :, :]
    w = np.nan_to_num(w, nan=1)
    return w

# Extract all labeled samples
X_all = []
y_all = []
for i, idx, lab in zip(orbits_idx, indices, labels):
    seq = prepare_seq(w[i], idx)
    X_all.append(seq)
    y_all.append(lab)

X_all = np.stack(X_all) # (N, 10, 1, 36, 36)
y_all = np.array(y_all)

X_good = X_all[y_all == 2]
X_bad = X_all[y_all != 2]

if 'X_good' in locals():
    # 1. Setup DataLoader
    # PyTorch default float is float32, Numpy is float64. We must cast.
    tensor_x = torch.tensor(X_good, dtype=torch.float32)
    
    dataloader = DataLoader(tensor_x, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE3D(latent_dim=LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Model initialized on {device}")

    # 3. Training Loop
    num_epochs = 100 
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, data in enumerate(dataloader):
            # FIX: Permute data to (Batch, Channel, Time, Height, Width)
            # Current: (Batch, Time, Channel, Height, Width) -> indices (0, 1, 2, 3, 4)
            # Target:  (Batch, Channel, Time, Height, Width) -> indices (0, 2, 1, 3, 4)
            data = data.permute(0, 2, 1, 3, 4).to(device)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f'Epoch {epoch+1}, Average Loss: {total_loss / len(X_good):.4f}')

#%%

    import numpy.ma as ma

    model.eval()
    with torch.no_grad():
        test_seq_np = X_good[0]
        test_seq = torch.tensor(test_seq_np, dtype=torch.float32).unsqueeze(0)
    
        # Permute for model
        test_seq = test_seq.permute(0, 2, 1, 3, 4).to(device)
    
        recon, mu, logvar = model(test_seq)
    
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 10, figsize=(20, 4))
    for i in range(1):
        k = i*2
        kk = k+1
        for j in range(10):        
            masked = ma.array(X_good[0, j, i, :, :], mask=(X_good[0, j, -1, :, :] == 0))
            axs[k, j].imshow(masked, vmin=0, vmax=1, cmap='Greys_r')
            axs[kk, j].imshow(recon[0, i, j, :, :], vmin=0, vmax=1, cmap='Greys_r')

#%%

    # 4. Evaluation
    score_good = []
    score_bad = []
    model.eval()
    with torch.no_grad():
        for seq_np in X_good:
            seq = torch.tensor(seq_np, dtype=torch.float32).unsqueeze(0)
            seq = seq.permute(0, 2, 1, 3, 4).to(device)
            recon, mu, logvar = model(seq)
            score_good.append(loss_function(recon, seq, mu, logvar).item())
        for seq_np in X_bad:
            seq = torch.tensor(seq_np, dtype=torch.float32).unsqueeze(0)
            seq = seq.permute(0, 2, 1, 3, 4).to(device)
            recon, mu, logvar = model(seq)
            score_bad.append(loss_function(recon, seq, mu, logvar).item())
        score_good = np.array(score_good)
        score_bad = np.array(score_bad)

#%%
else:
    print("X_all not found in memory. Run the data loading cells first.")

#%%

plt.figure()
plt.hist(score_bad, density=True, bins=100)
plt.hist(score_good, density=True, bins=100)

#%%

thres = 305

X_bb = X_bad[score_bad > thres]
X_gb = X_bad[score_bad <= thres]

idx = np.random.randint(0, X_bb.shape[0], 10)
fig, axs = plt.subplots(10, 10)
for i in range(10):
    for j in range(10):        
        axs[i, j].imshow(X_bb[idx[i], j, 0], vmin=0, vmax=1, cmap='Greys_r')
plt.suptitle('X_bb')

idx = np.random.randint(0, X_gb.shape[0], 10)
fig, axs = plt.subplots(10, 10)
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(X_gb[idx[i], j, 0], vmin=0, vmax=1, cmap='Greys_r')        
plt.suptitle('X_gb')







