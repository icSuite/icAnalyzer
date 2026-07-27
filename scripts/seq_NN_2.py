#%% Imports

import numpy as np
import glob
from icreader import ConductanceImage
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

#%% Paths

p_in = '/home/bing/Dropbox/work/data/conductance/'

#%% Fetch orbits available in all nc files

o = [int(o[-7:-3]) for o in sorted(glob.glob(p_in + '*.nc'))]

#%% Load all data

H, P, dH, dP = [], [], [], []
for orbit in tqdm(o, total=len(o)):
    filename = p_in + f'or_{str(orbit).zfill(4)}.nc'
    cI = ConductanceImage(filename)
    H.append(cI.H)
    P.append(cI.P)
    dH.append(cI.dH)
    dP.append(cI.dP)

#%% Load training data indices

q = np.load('/home/bing/Dropbox/work/temp_storage/seq_classification_results.npz')
orbits = q['orbits']
indices = q['indices']
labels = q['labels']

#%% Prepare sequences with mask

def prepare_seq(H_, P_, dH_, dP_, idx):
    h = H_[idx:idx+10]
    p = P_[idx:idx+10]
    dh = dH_[idx:idx+10]
    dp = dP_[idx:idx+10]
    mask = (~np.isnan(h) & ~np.isnan(p) & ~np.isnan(dh) & ~np.isnan(dp)).astype(np.float32)

    h = np.nan_to_num(h, nan=0.0)
    p = np.nan_to_num(p, nan=0.0)
    dh = np.nan_to_num(dh, nan=0.0)
    dp = np.nan_to_num(dp, nan=0.0)

    seq = np.stack([h, p, dh, dp, mask], axis=1)  # (10, 6, 36, 36)
    return seq

#%% Extract labeled training samples

X, y = [], []
for i, idx, lab in zip(orbits, indices, labels):
    seq = prepare_seq(H[i], P[i], dH[i], dP[i], idx)
    X.append(seq)
    y.append(lab)

X = np.stack(X)  # (N, 10, 5, 36, 36)
y = np.array(y)

# Normalize physical channels only (0:4)
phys_channels = X[:, :, :4, :, :]
mean = phys_channels.mean()
std = phys_channels.std()
X[:, :, :4, :, :] = (phys_channels - mean) / std

#%% Train/test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Data loaders

train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=32)

#%% Model

class MaskedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(50, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = x.reshape(x.size(0), -1, 36, 36)  # (B, 50, 36, 36)
        x = x.unsqueeze(2)  # (B, 50, 1, 36, 36)
        return self.fc(self.conv(x))

#%% Train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MaskedCNN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(100):
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

#%% Predict all orbits

predicted_good = []
predicted_bad = []
model.eval()
with torch.no_grad():
    for i in tqdm(range(len(o))):
        N = len(H[i])
        if N < 10:
            continue
        for start in range(N - 9):
            seq = prepare_seq(H[i], P[i], dH[i], dP[i], start)
            seq[:, :4, :, :] = (seq[:, :4, :, :] - mean) / std
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(seq_tensor)
            label = pred.argmax(dim=1).item()
            if label == 1:
                predicted_good.append((i, start))
            else:
                predicted_bad.append((i, start))

#%% Plot example
'''
i, start = predicted_good[0]
fig, axs = plt.subplots(2, 5, figsize=(15, 6))
for j in range(10):
    im = axs[j//5, j%5].imshow(H[i][start+j], cmap='inferno', vmin=0, vmax=60)
    axs[j//5, j%5].set_title(f'Time {j}')
fig.colorbar(im, ax=axs.ravel().tolist())
plt.tight_layout()
plt.show()
'''

#%%

np.save('/home/bing/Dropbox/work/code/repos/icAnalyzer/data/predicted_good.npy', np.array(predicted_good))
np.save('/home/bing/Dropbox/work/code/repos/icAnalyzer/data/predicted_bad.npy', np.array(predicted_bad))

#%% Plot good examples

ii = np.random.randint(0, len(predicted_good) - 1, 10)
fig, axs = plt.subplots(10, 10, figsize=(15,15))
for iax, i in enumerate(ii):
    orbit, idt = predicted_good[i]
    for jax in range(10):
        axs[iax, jax].imshow(P[orbit][idt+jax], vmin=0, vmax=10)
for ax in axs.flatten():
    ax.axis('off')
    
#%% Plot bad examples

ii = np.random.randint(0, len(predicted_bad) - 1, 10)
fig, axs = plt.subplots(10, 10, figsize=(15,15))
for iax, i in enumerate(ii):
    orbit, idt = predicted_bad[i]
    for jax in range(10):
        axs[iax, jax].imshow(P[orbit][idt+jax], vmin=0, vmax=10)
for ax in axs.flatten():
    ax.axis('off')
















