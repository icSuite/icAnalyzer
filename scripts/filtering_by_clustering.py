#%% Import 

import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from polplot import pp
from tqdm import tqdm
import scipy
from icreader import ConductanceImage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator

#%% Paths

p_in = '/home/bing/Dropbox/work/data/conductance/'

#%% Fetch orbits available in all nc files

# Fetch all orbits
o = [int(o[-7:-3]) for o in sorted(glob.glob(p_in + '*.nc'))]

#%% Load all images

H  = []
for orbit in tqdm(o, total=len(o)):
    filename = p_in + f'or_{str(orbit).zfill(4)}.nc'
    cI = ConductanceImage(filename)
    H.append(cI.wic_avg)
H  = np.concatenate(H, axis=0)

#%% Flatpack and impute

flat_data = H.reshape((H.shape[0], -1))  # (N, 1296)
flat_data[np.isnan(flat_data)] = 0#0

#%% Scale data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(flat_data)

#%% PCA

pca = PCA(n_components=X_scaled.shape[1])
components = pca.fit_transform(X_scaled)
#components = pca.fit_transform(flat_data)

#%% PCs

total = np.sum(pca.singular_values_)
individual = pca.singular_values_ / total * 100
cumulative = np.cumsum(pca.singular_values_ / total * 100)

fig, axs = plt.subplots(6, 6, figsize=(10,10))
for i, ax in enumerate(axs.flatten()):
   ax.imshow(pca.components_[i].reshape(36,36))
   ax.set_title(f'PC {i} - {np.round(individual[i], 1)} - {np.round(cumulative[i], 1)}')
   ax.axis('off')

#%% PC vs infomration

x = np.arange(1, X_scaled.shape[1]+1)
knee = KneeLocator(x, individual, curve='convex', direction='decreasing')

plt.figure()
plt.plot(x, individual)
plt.vlines(knee.knee, plt.gca().get_ylim()[0], plt.gca().get_ylim()[1], color='tab:red')
plt.title(f'Knee at {knee.knee}')
ax = plt.twinx()
ax.plot(x, cumulative, color='tab:orange')

#%% 

fig, axs = plt.subplots(3,3, figsize=(10,10))
axs = axs.flatten()
for ax, stop in zip(axs, [1, 3, 5, 10, 20, 40, 50, 73, 36*36-1]):
    pc_stop = pca.components_[:stop].T.dot(pca.singular_values_[:stop])
    pc_stop *= scaler.scale_
    pc_stop += scaler.mean_
    im = ax.imshow(pc_stop.reshape(36,36))
    fig.colorbar(im, ax=ax)
    ax.set_title(f'First {stop} PCs')
    ax.axis('off')

#%%

# Pick a real sample (e.g. the mean of the dataset)
mean_proj = np.zeros(X_scaled.shape[1])  # corresponds to mean because PCA is centered
sample_proj = components[0]              # or any real sample

fig, axs = plt.subplots(3, 3, figsize=(10, 10))
axs = axs.flatten()

for ax, stop in zip(axs, [1, 3, 5, 10, 20, 40, 50, 73, 36*36 - 1]):
    # Project using first `stop` PCs
    reconstruction_scaled = sample_proj[:stop] @ pca.components_[:stop, :]
    
    # Bring back to original scale
    reconstruction = reconstruction_scaled * scaler.scale_ + scaler.mean_

    # Plot
    im = ax.imshow(reconstruction.reshape(36, 36))
    fig.colorbar(im, ax=ax)
    ax.set_title(f'First {stop} PCs')
    ax.axis('off')

#%%

import umap

#reducer = umap.UMAP(n_components=2)#, random_state=42)
reducer = umap.UMAP(n_components=3)
X_umap = reducer.fit_transform(X_scaled)

#%%

fig, axs = plt.subplots(3, 1, figsize=(10, 10))
axs[0].scatter(X_umap[:, 0], X_umap[:, 1], s=.1, alpha=0.5)
axs[0].set_title("UMAP projection (2D)")
axs[0].set_xlabel("UMAP-1")
axs[0].set_ylabel("UMAP-2")
axs[1].scatter(X_umap[:, 0], X_umap[:, 2], s=.1, alpha=0.5)
axs[1].set_title("UMAP projection (2D)")
axs[1].set_xlabel("UMAP-1")
axs[1].set_ylabel("UMAP-3")
axs[2].scatter(X_umap[:, 1], X_umap[:, 2], s=.1, alpha=0.5)
axs[2].set_title("UMAP projection (2D)")
axs[2].set_xlabel("UMAP-2")
axs[2].set_ylabel("UMAP-3")


#%%

import hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=4000, min_samples=17)
labels = clusterer.fit_predict(X_umap)  # X_umap: your UMAP 2D/3D array

#%%

labels_ = labels + 1

# Assuming `labels` is a list or array of integers representing the label indices
# and `label_names` is a list of strings representing the label names
unique_labels = np.unique(labels_)
label_names = [f"Label {i}" for i in unique_labels]  # Replace with actual label names if available

# Create a colormap
cmap = plt.cm.get_cmap('tab10', len(unique_labels))
colors = cmap(unique_labels)

fig, axs = plt.subplots(3, 1, figsize=(10, 10))

# Plot the first scatter plot
scatter1 = axs[0].scatter(X_umap[:, 0], X_umap[:, 1], c=labels_, cmap='tab10', s=.1, alpha=0.5)
axs[0].set_title("UMAP projection (2D)")
axs[0].set_xlabel("UMAP-1")
axs[0].set_ylabel("UMAP-2")

# Plot the second scatter plot
scatter2 = axs[1].scatter(X_umap[:, 0], X_umap[:, 2], c=labels_, cmap='tab10', s=.1, alpha=0.5)
axs[1].set_title("UMAP projection (2D)")
axs[1].set_xlabel("UMAP-1")
axs[1].set_ylabel("UMAP-3")

# Plot the third scatter plot
scatter3 = axs[2].scatter(X_umap[:, 1], X_umap[:, 2], c=labels_, cmap='tab10', s=.1, alpha=0.5)
axs[2].set_title("UMAP projection (2D)")
axs[2].set_xlabel("UMAP-2")
axs[2].set_ylabel("UMAP-3")

# Create a custom legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in range(len(unique_labels))]
axs[0].legend(handles, label_names, title="Labels", loc="best")

plt.tight_layout()
plt.show()

#%%
plt.ioff()
for l in np.sort(np.unique(labels)):
    f = labels == l
    nl = np.sum(f)
    ii = np.random.uniform(0, nl-1, 5*5).astype(int)
    d = flat_data[labels == l][ii]
    fig, axs = plt.subplots(5, 5, figsize=(20, 20))
    for (di, ax) in zip(d, axs.flatten()):
        ax.imshow(di.reshape(36,36))
        ax.set_title(f'{l}' + f' : {np.sum(labels==l)}')
        ax.axis('off')
    plt.savefig(f'/home/bing/Dropbox/work/temp_storage/test_cluster/{l}.png', bbox_inches='tight')
    plt.close('all')
plt.ion()
    
#%%

from sklearn.metrics import silhouette_score

for mcs in [100, 200, 300]:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=5)
    labels = clusterer.fit_predict(X_umap)
    sil = silhouette_score(X_umap[labels != -1], labels[labels != -1])
    print(f"min_cluster_size={mcs}: silhouette={sil:.3f}, clusters={len(np.unique(labels))}")


#%%

from sklearn.metrics import calinski_harabasz_score

mcs = np.arange(20, 220, 20)
ms  = np.arange(2, 22, 2)
score = np.zeros((mcs.size, ms.size))

for i, mcsi in tqdm(enumerate(mcs), total = mcs.size):
    for j, msi in enumerate(ms):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=mcsi, min_samples=msi)
        labels = clusterer.fit_predict(X_umap)
        score[i, j] = calinski_harabasz_score(X_umap[labels != -1], labels[labels != -1])

#%%

from mpl_toolkits.mplot3d import Axes3D

X, Y = np.meshgrid(mcs, ms)   # Create a 2D grid from x and y

# Create the 3D surface plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, np.log10(score), cmap='viridis', edgecolor='none')

# Add color bar for reference
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# Set labels
ax.set_xlabel('mcs')
ax.set_ylabel('ms')
ax.set_zlabel('score')
ax.set_title('Tuning')

# Show the plot
plt.show()


#%%

# Parameters
mcs = np.arange(20, 220, 20)  # min_cluster_size
ms  = np.arange(2, 22, 2)     # min_samples

score = np.full((mcs.size, ms.size), np.nan)
n_clusters = np.zeros_like(score)
noise_frac = np.zeros_like(score)

# Grid search
for i, mcsi in tqdm(enumerate(mcs), total=mcs.size):
    for j, msi in enumerate(ms):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=mcsi, min_samples=msi)
        labels = clusterer.fit_predict(X_umap)

        if np.unique(labels).size > 1 and np.any(labels != -1):
            valid = labels != -1
            score[i, j] = calinski_harabasz_score(X_umap[valid], labels[valid])
            n_clusters[i, j] = len(set(labels)) - (1 if -1 in labels else 0)
            noise_frac[i, j] = np.mean(labels == -1)

#%%

from mpl_toolkits.mplot3d import Axes3D

X, Y = np.meshgrid(ms, mcs)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, np.log10(score), cmap='viridis', edgecolor='none')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

ax.set_xlabel('min_samples')
ax.set_ylabel('min_cluster_size')
ax.set_zlabel('log10(CH Score)')
ax.set_title('Calinski-Harabasz Score Surface (log scale)')
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(np.log10(n_clusters), aspect='auto', cmap='plasma', origin='lower',
           extent=[ms.min(), ms.max(), mcs.min(), mcs.max()])
plt.colorbar(label='Number of Clusters')
plt.xlabel('min_samples')
plt.ylabel('min_cluster_size')
plt.title('Number of Clusters Detected')
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(noise_frac, aspect='auto', cmap='inferno', origin='lower',
           extent=[ms.min(), ms.max(), mcs.min(), mcs.max()])
plt.colorbar(label='Noise Fraction')
plt.xlabel('min_samples')
plt.ylabel('min_cluster_size')
plt.title('Fraction of Points Marked as Noise')
plt.show()




#%%

from sklearn.cluster import DBSCAN

# Example DBSCAN on reduced data
db = DBSCAN(eps=0.5, min_samples=5).fit(X_umap)
labels = db.labels_

# Plot with cluster coloring
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=labels, palette='tab10', s=5, legend='full')
plt.title("UMAP + DBSCAN Clusters")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()



#%%



# Reduce dimensions to 50 or fewer
X_reduced = PCA(n_components=50).fit_transform(X_scaled)

#%% Data distribution

fig, axs = plt.subplots(3,3, figsize=(10,10))
for i, ax in enumerate(axs.flatten()):
    ii = int(np.random.uniform(0, 1250))
    ax.hist(flat_data[:, ii], bins=100)

#%% Cluster

nc = 11
kmeans = KMeans(n_clusters=nc, random_state=0)
labels = kmeans.fit_predict(flat_data)

#%% Print grouping

unique, counts = np.unique(labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Cluster {u}: {c} images")

#%% Plots clusters

# Cluster centroids: already in flat space
centroids = kmeans.cluster_centers_

fig, axs = plt.subplots(3, 4, figsize=(10,10))
for i, ax in enumerate(axs.flatten()[:nc]):  # one for each cluster
    img = centroids[i].reshape((36, 36))    
    im = ax.imshow(img, cmap='viridis')
    ax.set_title(f"Cluster {i} Centroid")
    cbar = fig.colorbar(im, ax=ax)
    
for i, ax in enumerate(axs.flatten()):  # one for each cluster
    ax.axis('off')

#%%

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Step 1: (Recommended) Normalize / scale the data
X_scaled = StandardScaler().fit_transform(flat_data)

# Step 2: Run DBSCAN
db = DBSCAN(eps=0.5, min_samples=2).fit(X_scaled)

# Step 3: View cluster labels
print("Labels:", db.labels_)

labels = db.labels_

# Number of clusters (excluding noise, which is -1)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# Number of noise points
n_noise = list(labels).count(-1)

print(f"Clusters found: {n_clusters}")
print(f"Noise points: {n_noise}")


# Optional: Plot
plt.scatter(flat_data, [0]*len(flat_data), c=db.labels_, cmap='tab10', s=100)
plt.yticks([])
plt.title("DBSCAN Clustering")
plt.show()


#%%

















#%%

# Assuming `data` is (N, 36, 36)
flat_data = H.reshape((H.shape[0], -1))  # (N, 1296)
valid_idx = ~np.isnan(flat_data).any(axis=1)
pca = PCA(n_components=36*36)
components = pca.fit_transform(flat_data[valid_idx])
#pca.singular_values_

flat_data[np.isnan(flat_data)] = 0

#%%

plt.figure()
plt.plot(pca.singular_values_/np.sum(pca.singular_values_)*100)
plt.plot(np.cumsum(pca.singular_values_/np.sum(pca.singular_values_)*100))


#%%
fig, axs = plt.subplots(6, 9, figsize=(9, 6))
for i, ax in enumerate(axs.flatten()):
    im = ax.imshow(pca.components_[i].reshape(36,36))
    cbar = fig.colorbar(im, ax=ax)    

#%%

mean_map = np.nanmean(H, axis=0)
std_map = np.nanstd(H, axis=0)
skew_map = scipy.stats.skew(H, axis=0, nan_policy='omit')

#%%

fig, axs = plt.subplots(3, 1, figsize=(10,10))
im = axs[0].imshow(mean_map)
axs[0].set_title('mean')
cbar = fig.colorbar(im, ax=axs[0])

im = axs[1].imshow(std_map)
axs[1].set_title('std')
cbar = fig.colorbar(im, ax=axs[1])

im = axs[2].imshow(skew_map)
cbar = fig.colorbar(im, ax=axs[2])
axs[2].set_title('skew')

#%%

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kneed import KneeLocator

inertias = []
k_range = np.arange(2, 35)
for k in tqdm(k_range, total=k_range.size):
    kmeans = KMeans(n_clusters=k, n_init='auto').fit(flat_data[valid_idx])
    inertias.append(kmeans.inertia_)

knee = KneeLocator(k_range, inertias, curve='convex', direction='decreasing')

plt.figure()
plt.plot(k_range, inertias, 'o-')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (within-cluster sum of squares)")
plt.title(f"Elbow Method for Optimal k: {knee.knee}")
plt.grid()
plt.show()

#%%


N = 11
rows

kmeans = KMeans(n_clusters=11, random_state=0)
labels = kmeans.fit_predict(flat_data[valid_idx])

unique, counts = np.unique(labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Cluster {u}: {c} images")


# Cluster centroids: already in flat space
centroids = kmeans.cluster_centers_

fig, axs = plt.subplots(3, 4)
for i, ax in enumerate(axs.flatten()):  # one for each cluster
    img = centroids[i].reshape((36, 36))    
    im = ax.imshow(img, cmap='viridis')
    ax.set_title(f"Cluster {i} Centroid")
    cbar = fig.colorbar(im, ax=ax)
    plt.axis('off')


#%%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import BayesianRidge  # or HistGradientBoostingRegressor

def iterative_impute_and_run_pca(flat_data_with_nans, n_components=10, plot=True, image_shape=(36, 36)):
    """
    Uses IterativeImputer to handle NaNs, then runs PCA.

    Parameters:
        flat_data_with_nans: ndarray of shape (N, D) with NaNs
        n_components: number of PCA components
        plot: whether to show plots
        image_shape: shape of original image (default 36x36)

    Returns:
        imputed_data: ndarray (N, D) with imputed values
        components: PCA-transformed data
        pca_model: trained PCA object
    """

    print("🔄 Imputing missing values using IterativeImputer...")
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
    imputed_data = imputer.fit_transform(flat_data_with_nans)

    print("🔍 Performing PCA on imputed data...")
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(imputed_data)

    if plot:
        # Explained variance plot
        plt.figure()
        plt.plot(np.arange(1, n_components+1), pca.explained_variance_ratio_, 'o-')
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance Ratio")
        plt.title("PCA Variance Explained")
        plt.grid()
        plt.show()

        # Spatial mode visualization
        for i in range(min(n_components, 5)):
            mode = pca.components_[i].reshape(image_shape)
            plt.figure()
            plt.imshow(mode, cmap='RdBu_r')
            plt.title(f"PCA Mode {i+1}")
            plt.colorbar()
            plt.axis('off')
            plt.show()

    return imputed_data, components, pca



#%%

# flat_data_with_nans: shape (450000, 1296), dtype float32/float64, with NaNs
imputed_data, pca_components, pca_model = iterative_impute_and_run_pca(flat_data[:10], n_components=10)



#%% Func

def get_c_scales(cI):
    c_scales = {'wicm': (0, np.round(np.nanmax(cI.wic_avg)+1)),
                'wics': (0, np.round(np.nanmax(cI.wic_std)+1)),
                's12m': (0, np.round(np.nanmax(cI.s12_avg)+1)),
                's12s': (0, np.round(np.nanmax(cI.s12_std)+1)),
                's13m': (0, np.round(np.nanmax(cI.s13_avg)+1)),
                's13s': (0, np.round(np.nanmax(cI.s13_std)+1)),
                'E0':   (0,  25),
                'dE0':  (0, np.round(np.nanmax(cI.dE0)+1)),
                'Fe':   (0, np.round(np.nanmax(cI.Fe)+1)),
                'dFe':  (0, np.round(np.nanmax(cI.dFe)+1)),
                'R':    (0, 150),
                'dR':   (0, np.round(5*np.median(cI.dR[~np.isnan(cI.dR)])+1)),
                'H':    (0, np.round(np.nanmax(cI.H)+1)),
                'dH':   (0, np.round(np.nanmax(cI.dH)+1)),
                'P':    (0, np.round(np.nanmax(cI.P)+1)),
                'dP':   (0, np.round(np.nanmax(cI.dP)+1))
                }
    return c_scales
    
def plot(cI, i, c_scales, lat, lt):    
        
    fig, axs = plt.subplots(3, 6, figsize=(30, 15))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    axes = axs.flatten()[:-2]
    axs[2, 4].set_axis_off()
    axs[2, 5].set_axis_off()
    
    var = [cI.wic_avg[i], cI.wic_std[i], cI.R[i],  cI.dR[i],  cI.H[i], cI.dH[i],
           cI.s13_avg[i], cI.s13_std[i], cI.E0[i], cI.dE0[i], cI.P[i], cI.dP[i],
           cI.s12_avg[i], cI.s12_std[i], cI.Fe[i], cI.dFe[i]]
    
    cs = [c_scales['wicm'], c_scales['wics'], c_scales['R'],  c_scales['dR'],  c_scales['H'], c_scales['dH'],
          c_scales['s13m'], c_scales['s13s'], c_scales['E0'], c_scales['dE0'], c_scales['P'], c_scales['dP'],
          c_scales['s12m'], c_scales['s12s'], c_scales['Fe'], c_scales['dFe']]
    
    tit = ['avg WIC counts', 'std WIC counts', 'WIC*/S13* (R)', 'R std', 'Hall', 'Hall std',
            'avg S13 counts', 'std S13 counts', 'E0', 'E0 std', 'Pedersen', 'Pedersen std',
            'avg S12 counts', 'std S12 counts', 'Fe', 'Fe std']
    
    for j, (ax, var_, cs_, tit_) in enumerate(zip(axes, var, cs, tit)):
        pax = pp(ax)
        if j == 12:
            pax.writeLTlabels(fontsize=16)
            ax.text(.85, .1, '50$^{\circ}$', ha='center', va='center', fontsize=16, transform=ax.transAxes)
        pax.plotimg(lat, lt, var_, crange=cs_)
        ax.set_title(tit_, fontsize=18)
        ax.text(.85, .85, str(int(cs_[-1])), ha='left', va='center', fontsize=16, transform=ax.transAxes)
    
    axs[0,2].text(1.1, 1.2, cI.time[i], ha='center', va='center', fontsize=20, transform=axs[0,2].transAxes)
    
#%% Plot

plt.ioff()
for orbit in tqdm(o, total=len(o)):
    filename = p_in + f'or_{str(orbit).zfill(4)}.nc'
    
    cI = ConductanceImage(filename)
    
    c_scales = get_c_scales(cI)
    
    lat = cI.grid.lat
    lt = (cI.grid.lon/15)%24
    
    p_out_o = p_out + f'or_{str(orbit).zfill(4)}/'
    os.makedirs(p_out_o, exist_ok=True)
    
    for i in range(cI.shape[0]):
        plot(cI, i, c_scales, lat, lt)
        plt.savefig(p_out_o + f'{str(i).zfill(3)}.png', bbox_inches='tight')
        plt.close('all')
