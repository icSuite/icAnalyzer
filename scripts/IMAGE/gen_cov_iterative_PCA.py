import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

import os
os.chdir('/home/bing/Dropbox/work/code/repos/icAnalyzer/scripts/IMAGE')
import helpers

# --- CONFIG ---
CONFIG = {
    'IMG_HEIGHT': 36, 'IMG_WIDTH': 36, 'N_MODES': 36,
    'MAX_ITER': 100, 'TOLERANCE': 1e-4, 'SPARSITY_THRESHOLD': 0.4,
    'N_DATA': 3000
}
PATHS = {
    'data_in': '/home/bing/Dropbox/work/data/IMAGE_FUV/conductance',
    'segments': '/home/bing/Dropbox/work/code/repos/icAnalyzer/data/good_orbit_segments.csv'
}


class IterativePCA:
    def __init__(self, n_modes=36, max_iter=50, tol=1e-4):
        self.n_modes = n_modes
        self.max_iter = max_iter
        self.tol = tol
        
    def fit_transform(self, X):
        """
        Assumes X is standardized (mean~0, std~1).
        """
        print(f"Iterative PCA (Linear Space) with {self.n_modes} modes...")
        
        mask = np.isnan(X)
        
        # 1. Initialize
        X_filled = X.copy()
        X_filled[mask] = 0.0 
        
        diffs = []
        for i in range(self.max_iter):
            # --- THE FIX: RE-CENTERING ---
            # As we fill values, the mean drifts. We must force it back to 0.
            # If we don't, PC1 becomes the "Mean Vector" and all correlations look positive.
            current_mean = np.mean(X_filled, axis=0)
            X_filled = X_filled - current_mean
            
            # 2. SVD
            U, S, Vt = scipy.linalg.svd(X_filled, full_matrices=False)
            
            # 3. Reconstruct
            X_recon = U[:, :self.n_modes] @ np.diag(S[:self.n_modes]) @ Vt[:self.n_modes, :]
            
            # 4. Update only missing values
            prev_filled = X_filled.copy()
            X_filled[mask] = X_recon[mask]
            
            # 5. Check Convergence
            diff = np.mean((X_filled[mask] - prev_filled[mask])**2)
            
            diffs.append(diff)
            print(f"  Iter {i+1}: MSE Change = {diff:.6f}")
            if diff < self.tol: break
            
        self.components_ = Vt[:self.n_modes, :]
        self.explained_variance_ = (S[:self.n_modes]**2) / (X.shape[0] - 1)
        
        return X_filled, diffs

if __name__ == "__main__":
    # 1. Load Data (Linear Mho)
    H, P, _, _, _ = helpers.load_data(PATHS, CONFIG['SPARSITY_THRESHOLD'])
    
    H_trunc, P_trunc = H[:CONFIG['N_DATA']], P[:CONFIG['N_DATA']]
    
    H_mu, H_sig = np.nanmean(H_trunc), np.nanstd(H_trunc)
    P_mu, P_sig = np.nanmean(P_trunc), np.nanstd(P_trunc)
    
    H_norm = (H_trunc - H_mu) / H_sig
    P_norm = (P_trunc - P_mu) / P_sig
    
    # No Log Transform!
    N = H_norm.shape[0]
    X = np.hstack([H_norm.reshape(N, -1), P_norm.reshape(N, -1)])
    
    # 2. Run IPCA
    ipca = IterativePCA(n_modes=CONFIG['N_MODES'], 
                        max_iter=CONFIG['MAX_ITER'])
    X_filled, diffs = ipca.fit_transform(X)
        
    plt.figure(figsize=(15, 9))
    plt.plot(diffs, label='Iteration difference')
    plt.hlines(ipca.tol, 0, len(diffs)-1, 'tab:red', '--', label='Threshold')
    plt.xlabel('Iteration')
    plt.ylabel('Diff')
    plt.title('Iterative PCA convergence')
    plt.legend()
    
    # 3. Stats
    print("Calculating Linear Covariance...")
    cov_matrix = np.cov(X_filled, rowvar=False)
    
    # 4. Plots (Same as VAE)
    plt.figure(figsize=(10, 8))
    plt.imshow(np.abs(cov_matrix[::4, ::4]), cmap='inferno', origin='upper')
    plt.colorbar(label='Covariance (Mho^2)')
    plt.title("Iterative PCA Covariance (Linear Space)")
    plt.show()
    
    # Teleconnections
    probe_idx = (18 * 36) + 6 
    corr_matrix = np.corrcoef(X_filled, rowvar=False)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    corr_h = corr_matrix[probe_idx, :36*36].reshape(36, 36)
    axs[0].imshow(corr_h, cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
    axs[0].scatter(18, 18, c='lime', marker='x', s=100)
    axs[0].set_title("IPCA Hall Correlation (Linear)")
    
    corr_p = corr_matrix[probe_idx, 36*36:].reshape(36, 36)
    axs[1].imshow(corr_p, cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
    axs[1].set_title("Cross-Correlation")
    plt.show()
    
    # Spectrum
    plt.figure()
    plt.plot(np.cumsum(ipca.explained_variance_) / np.sum(ipca.explained_variance_) * 100)
    plt.title("IPCA Explained Variance (Linear)")
    plt.show()