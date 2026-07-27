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
import pandas as pd

# Try importing custom library
try:
    from icreader import ConductanceImage
except ImportError:
    print("Warning: 'icreader' library not found. Data loading will fail unless fixed.")

# --- Hyperparameters ---
CONFIG = {
    'CHANNELS': 3,           # Input: Hall, Pedersen, Weight
    'IMG_HEIGHT': 36,
    'IMG_WIDTH': 36,
    'LATENT_DIM': 36,
    'BATCH_SIZE': 8,
    'LR': 1e-4,
    'EPOCHS': 100,
    'SEED': 42,
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

#%% 2. MODEL ARCHITECTURE (3-Channel Encoder, 2-Channel Decoder)

class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.multi_channel = kwargs.pop('multi_channel', False)
        self.return_mask = kwargs.pop('return_mask', False)
        super(PartialConv2d, self).__init__(*args, **kwargs)
        self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
        self.slide_winsize = self.kernel_size[0] * self.kernel_size[1]
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

        # --- ENCODER (Input: Hall, Pedersen, Weight) ---
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
        
        # Output: 2 Channels (Hall, Pedersen)
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
    # x: [B, 3, H, W] -> (H, P, W)
    # recon: [B, 2, H, W] -> (Rec_H, Rec_P)
    
    target = x[:, :2, :, :]
    weights = x[:, 2:3, :, :] # Huber Weight
    
    # Combined Weight: Mask * HuberWeight
    # This ensures we prioritize the auroral oval (high weight)
    final_weight = mask * weights
    
    recon_loss = torch.sum(final_weight * (recon - target)**2)
    # Normalize by the sum of weights (Stabilizes gradients)
    recon_loss = recon_loss / (final_weight.sum() + 1e-8) * (36*36*2)

    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld_loss

#%% 3. ANALYSIS TOOLS (Coupled Statistics, Adaptive OI, Ensemble)

def generate_coupled_log_statistics(model, latent_dim, H_stats, P_stats, n_samples=5000, epsilon=1.0):
    """Generates joint statistics for Hall and Pedersen."""
    H_mu, H_sig = H_stats
    P_mu, P_sig = P_stats
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim).to(device)
        samples_norm = model.decode(z) 
        
        # De-Normalize
        h_phys = np.maximum((samples_norm[:, 0, :, :].cpu().numpy() * H_sig) + H_mu, 0)
        p_phys = np.maximum((samples_norm[:, 1, :, :].cpu().numpy() * P_sig) + P_mu, 0)
        
        # Log Transform & Stack
        h_log = np.log(h_phys + epsilon).reshape(n_samples, -1)
        p_log = np.log(p_phys + epsilon).reshape(n_samples, -1)
        joint_state = np.hstack([h_log, p_log])
        
        mean_log = np.mean(joint_state, axis=0)
        cov_log = np.cov(joint_state, rowvar=False)
        return mean_log, cov_log

def analyze_explained_variance(cov_matrix, plot=True):
    """Calculates cumulative variance to determine 'k' modes."""
    vals, _ = scipy.linalg.eigh(cov_matrix)
    vals = vals[::-1] # Descending
    total_var = np.sum(vals)
    cum_var = np.cumsum(vals) / total_var
    
    n_90 = np.searchsorted(cum_var, 0.90) + 1
    n_95 = np.searchsorted(cum_var, 0.95) + 1
    n_99 = np.searchsorted(cum_var, 0.99) + 1
    
    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(cum_var * 100, label='Cumulative Variance')
        plt.axhline(95, color='r', linestyle='--', label=f'95% ({n_95} modes)')
        plt.title('Joint Covariance Spectrum')
        plt.xlabel('Modes')
        plt.ylabel('Variance Explained (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    return n_90, n_95, n_99

def generate_ensemble_oi(log_mean, log_cov, sparse_data_norm, sparse_mask, 
                         H_stats, P_stats, n_ensemble=10, error_std_phys=0.5, 
                         epsilon=1.0, target_variance=0.95):
    """
    Performs OI with ADAPTIVE TRUNCATION and generates an Ensemble + Trust Metric.
    """
    H_mu, H_sig = H_stats
    P_mu, P_sig = P_stats
    h, w = 36, 36
    N_pixels = h * w 
    
    # --- 1. PREPARE OBSERVATIONS ---
    y_h_norm = sparse_data_norm[0, 0].cpu().numpy().flatten()
    mask_h = sparse_mask[0, 0].cpu().numpy().flatten()
    y_p_norm = sparse_data_norm[0, 1].cpu().numpy().flatten()
    mask_p = sparse_mask[0, 1].cpu().numpy().flatten() 
    
    idx_h = np.where((mask_h > 0) & (~np.isnan(y_h_norm)))[0]
    idx_p = np.where((mask_p > 0) & (~np.isnan(y_p_norm)))[0]
    
    # --- 2. CONVERT TO LOG-PHYSICAL ---
    y_h_phys = np.maximum((y_h_norm[idx_h] * H_sig) + H_mu, 0)
    y_h_log = np.log(y_h_phys + epsilon)
    y_p_phys = np.maximum((y_p_norm[idx_p] * P_sig) + P_mu, 0)
    y_p_log = np.log(y_p_phys + epsilon)
    y_log = np.concatenate([y_h_log, y_p_log])
    
    # --- 3. BUILD INDICES ---
    state_idx_h = idx_h
    state_idx_p = idx_p + N_pixels
    valid_idx = np.concatenate([state_idx_h, state_idx_p])
    
    n_measurements = len(valid_idx)
    
    if n_measurements == 0:
        prior_ens = [((np.exp(log_mean[:N_pixels]).reshape(h,w)-epsilon), 
                      (np.exp(log_mean[N_pixels:]).reshape(h,w)-epsilon)) for _ in range(n_ensemble)]
        return prior_ens, np.zeros((h,w)), np.zeros((h,w))
    
    # --- 4. REDUCED RANK COVARIANCE (Adaptive) ---
    log_cov = np.nan_to_num(log_cov, nan=0.0)
    vals, vecs = scipy.linalg.eigh(log_cov)
    idx_sort = np.argsort(vals)[::-1]
    vals, vecs = vals[idx_sort], vecs[:, idx_sort]
    
    # A. Variance Requirement
    cum_var = np.cumsum(vals) / np.sum(vals)
    n_modes_var = np.searchsorted(cum_var, target_variance) + 1
    
    # B. Data Constraint (Effective DoF)
    n_modes_data = n_measurements
    
    # Take stricter limit
    n_modes = min(n_modes_var, n_modes_data)
    n_modes = max(n_modes, 1) # At least 1 mode
    n_modes = min(n_modes, len(vals)) # At most all modes
    
    print(f"Adaptive Truncation: {n_measurements} obs. Selected {n_modes} modes.")
    
    vals_k = vals[:n_modes]
    vecs_k = vecs[:, :n_modes]
    P_k = (vecs_k * vals_k) @ vecs_k.T
    
    prior_var = np.diag(P_k)
    
    # --- 5. SOLVE OI ---
    sigma_log_h = np.maximum(error_std_phys / (y_h_phys + epsilon), 0.05)
    sigma_log_p = np.maximum(error_std_phys / (y_p_phys + epsilon), 0.05)
    R = np.diag(np.concatenate([sigma_log_h, sigma_log_p])**2)
    
    x_b = log_mean
    P_Ht = P_k[:, valid_idx]
    H_P_Ht = P_k[np.ix_(valid_idx, valid_idx)]
    
    S = H_P_Ht + R + np.eye(len(valid_idx)) * 1e-6
    K = scipy.linalg.solve(S, P_Ht.T).T 
    
    innovation = y_log - x_b[valid_idx]
    x_analysis_log = x_b + K @ innovation
    
    # --- 6. TRUST METRIC ---
    explained_var = np.sum(K * P_Ht, axis=1)
    trust_metric = explained_var / (prior_var + 1e-9)
    trust_metric = np.clip(trust_metric, 0, 1)
    
    trust_h = trust_metric[:N_pixels].reshape(h, w)
    trust_p = trust_metric[N_pixels:].reshape(h, w)
    
    # --- 7. POSTERIOR COVARIANCE & SAMPLING ---
    correction = K @ P_Ht.T
    P_analysis = P_k - correction
    P_analysis = (P_analysis + P_analysis.T) / 2.0
    
    try:
        L = np.linalg.cholesky(P_analysis + np.eye(P_analysis.shape[0]) * 1e-8)
    except np.linalg.LinAlgError:
        evals, evecs = scipy.linalg.eigh(P_analysis)
        evals = np.maximum(evals, 0)
        L = evecs @ np.diag(np.sqrt(evals))
        
    ensemble_results = []
    for _ in range(n_ensemble):
        noise = np.random.randn(len(x_b))
        x_sample = x_analysis_log + L @ noise
        
        h_phys = np.maximum(np.exp(x_sample[:N_pixels]) - epsilon, 0).reshape(h, w)
        p_phys = np.maximum(np.exp(x_sample[N_pixels:]) - epsilon, 0).reshape(h, w)
        ensemble_results.append((h_phys, p_phys))
        
    return ensemble_results, trust_h, trust_p

#%% 4. MAIN EXECUTION

if __name__ == "__main__":
    
    # --- 4.1 Load Data ---
    gos = pd.read_csv(PATHS['segments'], header=None)
    gos.columns = ['orbit', 'shit_start', 'shit_stop', 'good_start', 'good_stop']
    gos = gos.loc[gos['orbit'] < 503, :]

    t, H, P, W = [], [], [], []
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
            W.append(cI.w[start:stop])
        except Exception as e:
            continue

    H = np.concatenate(H, axis=0)
    P = np.concatenate(P, axis=0)
    W = np.concatenate(W, axis=0)
    
    h, w = CONFIG['IMG_HEIGHT'], CONFIG['IMG_WIDTH']
    f = np.sum((np.isnan(H) | np.isnan(P)), axis=(1,2)) / (h*w)
    f = f < .33
    H, P, W = H[f], P[f], W[f]

    H_mu, H_sig = np.nanmean(H), np.nanstd(H)
    P_mu, P_sig = np.nanmean(P), np.nanstd(P)
    
    H_norm = (H - H_mu) / H_sig
    P_norm = (P - P_mu) / P_sig
    
    subset_size = 3000
    H_trunc, P_trunc, W_trunc = H_norm[:subset_size], P_norm[:subset_size], W[:subset_size]
    
    # Mask & Stack
    mask_np = (~(np.isnan(H_trunc) | np.isnan(P_trunc))).astype(np.float32)
    X_np = np.stack([np.nan_to_num(H_trunc, nan=0.0), 
                     np.nan_to_num(P_trunc, nan=0.0),
                     np.nan_to_num(W_trunc, nan=0.0)], axis=1)
    
    tensor_x = torch.from_numpy(X_np).float()
    tensor_mask = torch.from_numpy(mask_np).float().unsqueeze(1)
    dataset = TensorDataset(tensor_x, tensor_mask)
    dataloader = DataLoader(dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)

    # --- 4.2 Train Model ---
    model = VAE2D(latent_dim=CONFIG['LATENT_DIM']).to(CONFIG['DEVICE'])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['LR'])
    
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
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataset):.4f}")

    # --- 4.3 Statistics & Spectrum ---
    print("Generating Coupled Statistics...")
    mean_log, cov_log = generate_coupled_log_statistics(
        model, CONFIG['LATENT_DIM'], (H_mu, H_sig), (P_mu, P_sig)
    )

    print("Analyzing Spectrum...")
    analyze_explained_variance(cov_log, plot=True)
    

    # --- 4.4 Test Scenarios Setup ---
    id_ = 3600
    H_flat, P_flat = H_norm[id_].flatten(), P_norm[id_].flatten()
    valid_indices_h = np.where(~np.isnan(H_flat))[0]
    valid_indices_p = np.where(~np.isnan(P_flat))[0]
    
    # Sample random points
    idx_h = np.random.choice(valid_indices_h, 50, replace=False)
    idx_p = idx_h
    #idx_p = np.random.choice(valid_indices_p, 300, replace=False)
    
    mask_h_base = np.zeros_like(H_flat)
    mask_p_base = np.zeros_like(P_flat)
    mask_h_base[idx_h] = 1.0
    mask_p_base[idx_p] = 1.0
    
    # Common Sparse Data (filled where mask is 1)
    H_sparse = np.nan_to_num(H_flat * mask_h_base, nan=0.0)
    P_sparse = np.nan_to_num(P_flat * mask_p_base, nan=0.0)
    sparse_data = torch.from_numpy(np.stack([H_sparse.reshape(36,36), P_sparse.reshape(36,36)])).float().unsqueeze(0)
    
    # --- 4.5 Comparison: Hall Only, Ped Only, Both ---
    print("Running Scenario Analysis...")
    
    # Scenario 1: Both Inputs
    mask_both = torch.from_numpy(np.stack([mask_h_base.reshape(36,36), mask_p_base.reshape(36,36)])).float().unsqueeze(0)
    ens_both, _, _ = generate_ensemble_oi(mean_log, cov_log, sparse_data, mask_both, (H_mu, H_sig), (P_mu, P_sig), n_ensemble=1)
    mean_both = ens_both[0] 
    
    # Scenario 2: Hall Only
    mask_hall = torch.from_numpy(np.stack([mask_h_base.reshape(36,36), np.zeros((36,36))])).float().unsqueeze(0)
    ens_hall, _, _ = generate_ensemble_oi(mean_log, cov_log, sparse_data, mask_hall, (H_mu, H_sig), (P_mu, P_sig), n_ensemble=1)
    mean_hall = ens_hall[0]
    
    # Scenario 3: Ped Only
    mask_ped = torch.from_numpy(np.stack([np.zeros((36,36)), mask_p_base.reshape(36,36)])).float().unsqueeze(0)
    ens_ped, _, _ = generate_ensemble_oi(mean_log, cov_log, sparse_data, mask_ped, (H_mu, H_sig), (P_mu, P_sig), n_ensemble=1)
    mean_ped = ens_ped[0]

    # Plot Comparison
    fig, axs = plt.subplots(5, 2, figsize=(10, 15), constrained_layout=True)
    gt_h = (H_norm[id_] * H_sig) + H_mu
    gt_p = (P_norm[id_] * P_sig) + P_mu
    vmax_h, vmax_p = np.nanmax(gt_h), np.nanmax(gt_p)
    
    # GT & Inputs
    axs[0,0].imshow(gt_h, cmap='plasma', origin='lower', vmin=0, vmax=vmax_h); axs[0,0].set_title("GT Hall")
    axs[0,1].imshow(gt_p, cmap='plasma', origin='lower', vmin=0, vmax=vmax_p); axs[0,1].set_title("GT Ped")
    
    axs[1,0].imshow(np.full((36,36), np.nan), cmap='gray', origin='lower'); axs[1,0].set_title("Input H")
    y, x = np.where(mask_h_base.reshape(36,36) > 0); axs[1,0].scatter(x, y, s=5, c='k')
    axs[1,1].imshow(np.full((36,36), np.nan), cmap='gray', origin='lower'); axs[1,1].set_title("Input P")
    y, x = np.where(mask_p_base.reshape(36,36) > 0); axs[1,1].scatter(x, y, s=5, c='k')

    # Reconstructions
    axs[2,0].imshow(mean_hall[0], cmap='plasma', origin='lower', vmin=0, vmax=vmax_h); axs[2,0].set_title("Rec (In: Hall)")
    axs[2,1].imshow(mean_hall[1], cmap='plasma', origin='lower', vmin=0, vmax=vmax_p); axs[2,1].set_title("Rec (In: Hall)")
    
    axs[3,0].imshow(mean_ped[0], cmap='plasma', origin='lower', vmin=0, vmax=vmax_h); axs[3,0].set_title("Rec (In: Ped)")
    axs[3,1].imshow(mean_ped[1], cmap='plasma', origin='lower', vmin=0, vmax=vmax_p); axs[3,1].set_title("Rec (In: Ped)")
    
    axs[4,0].imshow(mean_both[0], cmap='plasma', origin='lower', vmin=0, vmax=vmax_h); axs[4,0].set_title("Rec (In: Both)")
    axs[4,1].imshow(mean_both[1], cmap='plasma', origin='lower', vmin=0, vmax=vmax_p); axs[4,1].set_title("Rec (In: Both)")
    plt.suptitle("Cross-Covariance Tests")
    plt.show()

    # --- 4.6 Ensemble & Trust Analysis ---
    print("Running Ensemble Analysis...")
    ensemble, trust_h, trust_p = generate_ensemble_oi(
        mean_log, cov_log, sparse_data, mask_both, 
        (H_mu, H_sig), (P_mu, P_sig), n_ensemble=10, target_variance=0.95
    )
    
    # Plot Trust and Uncertainty
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    # Variance
    stack_h = np.stack([e[0] for e in ensemble])
    axs[0,0].imshow(np.var(stack_h, axis=0), cmap='viridis', origin='lower')
    axs[0,0].set_title("Ensemble Variance (H)")
    
    stack_p = np.stack([e[1] for e in ensemble])
    axs[0,1].imshow(np.var(stack_p, axis=0), cmap='viridis', origin='lower')
    axs[0,1].set_title("Ensemble Variance (P)")
    
    # Trust
    im = axs[1,0].imshow(trust_h, cmap='RdYlGn', vmin=0, vmax=1, origin='lower')
    axs[1,0].set_title("Trust Metric (H)")
    plt.colorbar(im, ax=axs[1,0])
    
    im = axs[1,1].imshow(trust_p, cmap='RdYlGn', vmin=0, vmax=1, origin='lower')
    axs[1,1].set_title("Trust Metric (P)")
    plt.colorbar(im, ax=axs[1,1])
    
    plt.suptitle("Uncertainty & Trust")
    plt.show()