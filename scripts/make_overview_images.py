#%% Import

import os
from os.path import join as pjoin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import glob
from tqdm import tqdm
from icreader import ConductanceImage

#%% Paths

base = '/home/bing/Dropbox/work'
#p_in = pjoin(base, 'data', 'IMAGE_FUV', 'conductance')
#p_out = pjoin(base, 'code', 'repos', 'icAnalyzer', 'figures', 'overview_wic')
p_in = pjoin(base, 'data', 'IMAGE_FUV', 'conductance_new')
p_out = pjoin(base, 'projects', 'icAnalyzer', 'overview_wic_new')

#%% Fetch orbits available in all nc files

# Fetch all orbits
o = [int(o[-7:-3]) for o in sorted(glob.glob(p_in + '*.nc'))]

#%% Plot every 10th image in each orbit for quick overview and orbit selection

plt.ioff()
for orbit in tqdm(o, total=len(o)):
    cI = ConductanceImage(pjoin(p_in, f'or_{str(orbit).zfill(4)}.nc'))
    wic = cI.wic_avg
    s12 = cI.s12_avg
    s13 = cI.s13_avg

    fig, axes = plt.subplots(6, 6, figsize=(20, 20))
    axs = axes.flatten()
    j = 0
    for i in np.arange(0, wic.shape[0], 10):
        axs[j].imshow(wic[i])
        j += 1
    if j != axs.size:
        for jj in range(j, axs.size):
            axs[jj].set_axis_off()
    
    plt.suptitle(f'Orbit {str(orbit).zfill(4)}')
    
    fileout = pjoin(p_out, f'orbit_{str(orbit).zfill(4)}.png')
    plt.savefig(fileout, bbox_inches='tight')
    plt.close('all')
plt.ion()

#%% Import not completly shitty orbits

filename = pjoin(base, 'code', 'repos', 'icAnalyzer', 'data', 'not_completely_shit_orbits.csv')
ncso = pd.read_csv(filename, header=None)
ncso.columns = ['orbit', 'good']
ncso = ncso.loc[ncso['good']==1].reset_index().drop(columns='index')

#%% Loop over all ncso and plot all figs in stacks of 50

p_out_ncso = pjoin(base, 'code', 'repos', 'icAnalyzer', 'figures', 'ncso_new')
os.makedirs(p_out_ncso, exist_ok=True)

col, row = 6, 6
n = col*row

plt.ioff()
for orbit in tqdm(ncso['orbit'], total=ncso.shape[0]):
    
    cI = ConductanceImage(pjoin(p_in, f'or_{str(orbit).zfill(4)}.nc'))
    wic = cI.wic_avg
    vmax = np.nanquantile(wic, .95)
    
    start = 0
    for i in range(int(np.ceil(wic.shape[0]/n))):
        fig, axes = plt.subplots(row, col, figsize=(20, 20))
        axs = axes.flatten()
        for ax in axs:
            ax.set_axis_off()
            
        for j, jj in enumerate(range(start, np.min((start+n, wic.shape[0])))):
            axs[j].imshow(wic[jj], vmin=0, vmax=vmax)
            axs[j].set_title(jj, fontsize=15)
        
        plt.suptitle(f'Orbit {str(orbit).zfill(4)}', y=.95, fontsize=30)
        
        fileout = pjoin(p_out_ncso, f'orbit_{str(orbit).zfill(4)}_{str(start).zfill(3)}.png')
        plt.savefig(fileout, bbox_inches='tight')
        plt.close('all')
            
        start += n
plt.ion()

#%%

filename = pjoin(base, 'code', 'repos', 'icAnalyzer', 'data', 'good_orbit_segments.csv')
gos = pd.read_csv(filename, header=None)
gos.columns = ['orbit', 'shit_start', 'shit_stop', 'good_start', 'good_stop']
#gos = gos.loc[gos['orbit']<233, :]
gos = gos.loc[gos['orbit']<503, :]

H, P = [], []
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
    
    H.append(cI.H[int(start):int(stop+1)])
    P.append(cI.P[int(start):int(stop+1)])
        
#%%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ---------------------------------------------------------
# 1. Setup Data (Assuming 'H' is your list of arrays)
# ---------------------------------------------------------
# Ensure H is your list: e.g. H = [array1, array2, ...]
# full_data = np.concatenate(H, axis=0) 

# --- FOR DEMO PURPOSES: Generating dummy data that mimics your error ---
# (Delete this block when running on your actual H list)
#data_list = [np.random.rand(100, 36, 36) for _ in range(3)]
full_data = np.concatenate(P, axis=0)
n_samples, h, w = full_data.shape
# Create a mask where corners are ALWAYS NaN (simulating your error source)
y, x = np.ogrid[:h, :w]
mask_circular = (x - w/2)**2 + (y - h/2)**2 > (w/2)**2
full_data[:, mask_circular] = np.nan 
# -----------------------------------------------------------------------

# Flatten the data
X = full_data.reshape(n_samples, h * w)

# ---------------------------------------------------------
# 2. Identify and Remove "Dead" Pixels (Always NaN)
# ---------------------------------------------------------
# Find columns (pixels) where NOT ALL values are NaN
# valid_pixel_mask is a boolean array of length 1296 (36*36)
valid_pixel_mask = ~np.all(np.isnan(X), axis=0)

# Subset X to keep only pixels that have at least some data
X_subset = X[:, valid_pixel_mask]

print(f"Original features: {X.shape[1]}")
print(f"Valid features (excluding permanent NaNs): {X_subset.shape[1]}")

# ---------------------------------------------------------
# 3. Iterative PCA on Valid Pixels
# ---------------------------------------------------------
# Identify where NaNs are in this subset (sporadic missing data)
nan_mask_subset = np.isnan(X_subset)

# Initial Mean Imputation (Seed)
col_mean = np.nanmean(X_subset, axis=0)
X_filled = np.where(nan_mask_subset, col_mean, X_subset)

# Center the data
X_centered = X_filled - col_mean

print("Running Iterative PCA on valid pixels...")
n_components = 20
pca = PCA(n_components=n_components)

# Iteration loop
n_iter = 100
for i in range(n_iter):
    pca.fit(X_centered)
    X_reconstructed = pca.inverse_transform(pca.transform(X_centered))
    # Update only the missing slots
    X_centered[nan_mask_subset] = X_reconstructed[nan_mask_subset]

print("Converged.")

# ---------------------------------------------------------
# 4. Reconstruct Full Images for Plotting
# ---------------------------------------------------------
# pca.components_ has shape (20, n_valid_pixels)
# We need to map this back to (20, 1296)

# Create an array of NaNs for the full shape
full_components = np.full((n_components, h * w), np.nan)

# Fill in the calculated components into their correct valid positions
full_components[:, valid_pixel_mask] = pca.components_

# Reshape to (20, 36, 36)
eigenimages = full_components.reshape(n_components, h, w)

# ---------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------
fig, axes = plt.subplots(4, 5, figsize=(12, 10))
fig.suptitle(f'Top {n_components} Eigenimages (Dead pixels excluded)', fontsize=16)

axes_flat = axes.flatten()
for i in range(n_components):
    ax = axes_flat[i]
    current_cmap = plt.cm.plasma
    current_cmap.set_bad(color='lightgrey') # Dead pixels show as grey
    
    im = ax.imshow(eigenimages[i], cmap=current_cmap, origin='lower')
    ax.set_title(f'PC {i+1} ({pca.explained_variance_ratio_[i]*100:.1f}%)')
    ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()

#%%    

# ---------------------------------------------------------
# Prerequisite: Ensure you have 'X_centered' and 'valid_pixel_mask'
# from the previous Iterative PCA script.
# ---------------------------------------------------------

# 1. Calculate Correlation Matrix
# rowvar=False tells numpy that rows are time steps, columns are pixels
corr_matrix = np.corrcoef(X_centered, rowvar=False)

print(f"Correlation Matrix shape: {corr_matrix.shape}") 
# Likely (N_valid_pixels, N_valid_pixels), e.g., (800, 800)

# ---------------------------------------------------------
# Visualization 1: The Full Matrix (Pixel-to-Pixel)
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
plt.colorbar(label='Correlation Coefficient')
plt.title('Pixel-Pixel Correlation Matrix')
plt.xlabel('Pixel Index (Flattened)')
plt.ylabel('Pixel Index (Flattened)')
plt.show()

# ---------------------------------------------------------
# Visualization 2: "Seed" Correlation Map (Physical View)
# ---------------------------------------------------------
# Pick a specific pixel coordinate to test (e.g., center of image)
seed_y, seed_x = 18, 18 

# Convert 2D (y, x) to flattened index
# We need to account for the fact that we removed "dead" pixels.
# We create a temporary 2D mask to find the index in the VALID subset.

# 1. Create a dummy map of indices
full_indices = np.arange(h * w).reshape(h, w)
target_flat_index = full_indices[seed_y, seed_x]

# 2. Check if the seed is valid (not a dead pixel)
if not valid_pixel_mask[target_flat_index]:
    print("Error: The chosen seed pixel is a 'dead' (always NaN) pixel.")
else:
    # 3. Find where this pixel lives in the SUBSET (X_centered)
    # np.cumsum(valid_pixel_mask) maps original indices to subset indices
    subset_index = np.cumsum(valid_pixel_mask)[target_flat_index] - 1
    
    # 4. Extract the correlation row for this pixel
    # This array contains the correlation of the seed vs all valid pixels
    seed_correlations_subset = corr_matrix[subset_index, :]
    
    # 5. Map back to full 36x36 grid for plotting
    full_corr_map = np.full((h * w), np.nan)
    full_corr_map[valid_pixel_mask] = seed_correlations_subset
    full_corr_map = full_corr_map.reshape(h, w)
    
    # Plot
    plt.figure(figsize=(6, 5))
    plt.imshow(full_corr_map, cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
    plt.colorbar(label='Correlation')
    
    # Mark the seed point
    plt.scatter([seed_x], [seed_y], c='black', marker='x', label='Seed Pixel')
    
    plt.title(f'Correlation Map relative to pixel ({seed_x}, {seed_y})')
    plt.legend()
    plt.show()

    
#%%    

# ---------------------------------------------------------
# Visualization 2: "Seed" Correlation Map (Physical View)
# ---------------------------------------------------------
# Pick a specific pixel coordinate to test (e.g., center of image)
seed_y, seed_x = 18, 27

# Convert 2D (y, x) to flattened index
# We need to account for the fact that we removed "dead" pixels.
# We create a temporary 2D mask to find the index in the VALID subset.

# 1. Create a dummy map of indices
full_indices = np.arange(h * w).reshape(h, w)
target_flat_index = full_indices[seed_y, seed_x]

# 2. Check if the seed is valid (not a dead pixel)
if not valid_pixel_mask[target_flat_index]:
    print("Error: The chosen seed pixel is a 'dead' (always NaN) pixel.")
else:
    # 3. Find where this pixel lives in the SUBSET (X_centered)
    # np.cumsum(valid_pixel_mask) maps original indices to subset indices
    subset_index = np.cumsum(valid_pixel_mask)[target_flat_index] - 1
    
    # 4. Extract the correlation row for this pixel
    # This array contains the correlation of the seed vs all valid pixels
    seed_correlations_subset = corr_matrix[subset_index, :]
    
    # 5. Map back to full 36x36 grid for plotting
    full_corr_map = np.full((h * w), np.nan)
    full_corr_map[valid_pixel_mask] = seed_correlations_subset
    full_corr_map = full_corr_map.reshape(h, w)
    
    # Plot
    plt.figure(figsize=(6, 5))
    plt.imshow(full_corr_map, cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
    plt.colorbar(label='Correlation')
    
    # Mark the seed point
    plt.scatter([seed_x], [seed_y], c='black', marker='x', label='Seed Pixel')
    
    plt.title(f'Correlation Map relative to pixel ({seed_x}, {seed_y})')
    plt.legend()
    plt.show() 

#%%

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Setup: Create Covariance Matrix (Reduced Space)
# ---------------------------------------------------------
# X_centered has shape (N_samples, 1007). 
# It contains NO dead pixels.
cov_matrix = np.cov(X_centered, rowvar=False)

# This is our "True" state in the reduced 1D space
true_state_reduced = X_centered[20] 

# ---------------------------------------------------------
# 2. Simulate Sparse Measurements
# ---------------------------------------------------------
# We select 15 random INDICES from the REDUCED vector (0 to 1006)
n_measurements = 50
obs_indices_reduced = np.random.choice(len(true_state_reduced), n_measurements, replace=False)

#rows, cols = np.arange(7, 29), np.arange(7, 29)
#obs_indices_reduced = ((rows-1)*35 + cols).astype(int)
#n_measurements = obs_indices_reduced.size

# The Observation Operator H (extracts specific pixels from reduced vector)
H_op = np.zeros((n_measurements, len(true_state_reduced)))
for i, idx in enumerate(obs_indices_reduced):
    H_op[i, idx] = 1

# Create observations (Truth + small noise)
obs_noise_std = 0.05
y_obs = true_state_reduced[obs_indices_reduced] + np.random.normal(0, obs_noise_std, n_measurements)

# ---------------------------------------------------------
# 3. The Reconstruction (Optimal Interpolation)
# ---------------------------------------------------------
# Background guess (xb): 0 (since data is centered)
xb = np.zeros_like(true_state_reduced)

# Error Covariance R
R = np.eye(n_measurements) * (obs_noise_std**2)

# --- Kalman Gain Calculation (All in Reduced Space) ---
# P H^T
P_HT = cov_matrix[:, obs_indices_reduced]

# H P H^T
H_P_HT = P_HT[obs_indices_reduced, :]

# Calculate Gain K
inv_term = np.linalg.inv(H_P_HT + R)
K = P_HT @ inv_term 

# Update: xa = xb + K(y - H xb)
innovation = y_obs - xb[obs_indices_reduced]
xa_reduced = xb + K @ innovation

# ---------------------------------------------------------
# 4. Helper Function: Inflate Reduced Data to Full Image
# ---------------------------------------------------------
def restore_image(reduced_data, mask, shape=(36, 36)):
    """
    Maps the 1D reduced data back to the 2D full grid.
    mask: boolean array of valid pixels (from previous script)
    """
    full_flat = np.full(np.prod(shape), np.nan) # Start with all NaNs
    full_flat[mask] = reduced_data              # Fill valid spots
    return full_flat.reshape(shape)

# ---------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------
h, w = 36, 36
# Restore the "True" state to 2D
image_truth = restore_image(true_state_reduced, valid_pixel_mask, (h, w))
# Restore the "Analysis" (Reconstruction) to 2D
image_analysis = restore_image(xa_reduced, valid_pixel_mask, (h, w))

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# A. Plot Truth
im0 = ax[0].imshow(image_truth, origin='lower', cmap='plasma')
ax[0].set_title("True Field (Hidden)")

# B. Plot Sparse Inputs
# We need to find where the reduced indices are in the FULL 2D space
# to plot the 'x' markers correctly.
# 'valid_pixel_mask' is boolean size 1296. 
# np.where(valid_pixel_mask)[0] gives the indices of valid pixels in full space.
valid_indices_full_space = np.where(valid_pixel_mask)[0]
obs_indices_full_space = valid_indices_full_space[obs_indices_reduced]

# Convert flat full indices to (x, y) coordinates
obs_y, obs_x = np.unravel_index(obs_indices_full_space, (h, w))

# Create a blank map for the sparse plot
sparse_map = np.full((h, w), np.nan)
# We can't easily imshow sparse points, so we'll just plot the markers
ax[1].imshow(np.full((h, w), np.nan), origin='lower') # Blank background
ax[1].set_facecolor('black') # distinct background
sc = ax[1].scatter(obs_x, obs_y, c=y_obs, cmap='plasma', s=100, edgecolors='white',
                   vmin=np.nanmin(image_truth), vmax=np.nanmax(image_truth))
ax[1].set_title(f"Sparse Inputs ({n_measurements} pts)")
plt.colorbar(sc, ax=ax[1], fraction=0.046, pad=0.04)

# C. Plot Reconstruction
im2 = ax[2].imshow(image_analysis, origin='lower', cmap='plasma', 
                   vmin=np.nanmin(image_truth), vmax=np.nanmax(image_truth))
ax[2].set_title("Reconstructed from Covariance")

plt.tight_layout()
plt.show()
    
#%%

# Calculate Analysis Error Covariance Matrix
# I is Identity matrix of size (N_pixels, N_pixels)
I = np.eye(len(true_state_reduced))
Pa = (I - K @ H_op) @ cov_matrix

# Extract uncertainty (Standard Deviation)
uncertainty_map_reduced = np.sqrt(np.diag(Pa))

# Visualize
uncertainty_map = restore_image(uncertainty_map_reduced, valid_pixel_mask)
plt.imshow(uncertainty_map, origin='lower', cmap='hot')
plt.title("Estimated Uncertainty Map")
plt.colorbar(label="Conductance Error (S)")    
    
#%%
# --- 1. Uncertainty Quantification (Analysis Error) ---

# Identity matrix for the reduced space (1007 x 1007)
I = np.eye(len(true_state_reduced))

# Calculate Analysis Error Covariance: Pa = (I - KH)P
# Note: For huge matrices, we usually avoid computing the full Pa, 
# but for 1000 pixels, this is instant.
Pa = (I - K @ H_op) @ cov_matrix

# The diagonal contains the variance (uncertainty^2) for each pixel
posterior_variance = np.diag(Pa)
uncertainty_map_reduced = np.sqrt(posterior_variance)

# Map back to 2D image
uncertainty_map = restore_image(uncertainty_map_reduced, valid_pixel_mask, (h, w))

# Background uncertainty (before data) for comparison
prior_uncertainty_map = restore_image(np.sqrt(np.diag(cov_matrix)), valid_pixel_mask, (h, w))    
    
#%%

# --- 2. Degrees of Freedom (Eigenvalues) ---

# Calculate eigenvalues of the Covariance Matrix
eigvals = np.linalg.eigvalsh(cov_matrix)

# Sort descending (largest variance modes first)
eigvals = eigvals[::-1] 

# Calculate cumulative explained variance
cumulative_variance = np.cumsum(eigvals)
total_variance = np.sum(eigvals)
explained_ratio = cumulative_variance / total_variance

# Find how many modes are needed to explain 90% and 99% of the variance
n_90 = np.argmax(explained_ratio > 0.90) + 1
n_99 = np.argmax(explained_ratio > 0.99) + 1

print(f"Total Degrees of Freedom (pixels): {len(eigvals)}")
print(f"Sensors needed for 90% accuracy: {n_90}")
print(f"Sensors needed for 99% accuracy: {n_99}")    
    
#%%    

# --- 3. Leave-One-Out Cross Validation (LOOCV) ---

errors = []

print(f"Running Cross-Validation on {n_measurements} points...")

for i in range(n_measurements):
    # --- A. Split Data ---
    # Hold out the i-th measurement as "Truth"
    test_idx = obs_indices_reduced[i]
    test_y = y_obs[i]
    
    # Use all OTHER measurements for Training
    train_indices = np.delete(obs_indices_reduced, i)
    train_y = np.delete(y_obs, i)
    
    # --- B. Build K for this subset ---
    # Slice covariance for the TRAINING points
    P_HT_train = cov_matrix[:, train_indices]
    H_P_HT_train = P_HT_train[train_indices, :]
    
    # Noise matrix for training set (1 smaller)
    R_train = np.eye(n_measurements - 1) * (obs_noise_std**2)
    
    # Calculate K_train
    inv_term = np.linalg.inv(H_P_HT_train + R_train)
    K_train = P_HT_train @ inv_term
    
    # --- C. Predict ---
    # Update state using training data
    innovation = train_y - 0  # xb is 0
    xa_cv = 0 + K_train @ innovation
    
    # Prediction at the HELD OUT location
    predicted_val = xa_cv[test_idx]
    
    # --- D. Store Error ---
    errors.append(predicted_val - test_y)

rmse = np.sqrt(np.mean(np.array(errors)**2))
print(f"Cross-Validation RMSE: {rmse:.4f}")
print(f"Observation Noise (Target): {obs_noise_std:.4f}")

if rmse < obs_noise_std * 1.5:
    print(">> FIT QUALITY: EXCELLENT (Error is dominated by sensor noise)")
elif rmse < obs_noise_std * 3:
    print(">> FIT QUALITY: ACCEPTABLE")
else:
    print(">> FIT QUALITY: POOR (Model cannot resolve these features)")

#%%

fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2, 2)

# Plot 1: Uncertainty Map
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(uncertainty_map, origin='lower', cmap='gray_r')
ax1.set_title("Uncertainty Map (Darker = Better)")
# Overlay sensors to show how uncertainty drops near them
obs_y, obs_x = np.unravel_index(np.where(valid_pixel_mask)[0][obs_indices_reduced], (h, w))
ax1.scatter(obs_x, obs_y, c='cyan', marker='x', label='Sensors')
plt.colorbar(im1, ax=ax1, label="Std Dev")

# Plot 2: Improvement (Prior vs Posterior Uncertainty)
# This shows exactly how much information the sensors added
ax2 = fig.add_subplot(gs[0, 1])
improvement = prior_uncertainty_map - uncertainty_map
im2 = ax2.imshow(improvement, origin='lower', cmap='Greens')
ax2.set_title("Information Gain (Reduction in Uncertainty)")
plt.colorbar(im2, ax=ax2, label="Uncertainty Reduction")

# Plot 3: Eigenvalue Spectrum (Degrees of Freedom)
ax3 = fig.add_subplot(gs[1, :])
ax3.plot(explained_ratio, 'k-', lw=2)
ax3.axhline(0.90, color='r', linestyle='--', label='90% Variance')
ax3.axvline(n_90, color='r', linestyle='--')
ax3.set_title("Cumulative Variance Explained by Covariance Modes")
ax3.set_xlabel("Number of Modes / Sensors")
ax3.set_ylabel("Cumulative Variance")
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()







