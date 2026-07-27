#%% Import 

import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from polplot import pp
from tqdm import tqdm
from icreader import ConductanceImage
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
from scipy.interpolate import griddata

#%% Paths

p_in = '/home/bing/Dropbox/work/data/conductance/'

base = '/home/bing/Dropbox/work/code/repos/icAnalyzer/'

#%% Load substorm list

# Read the CSV with datetime parsing
sl = pd.read_csv(base + 'data/newell_substorm_list.csv', parse_dates=['Date_UTC'])
slt = np.array([datetime(t.year, t.month, t.day, t.hour, t.minute, t.second) for t in sl['Date_UTC']])
slmlt = sl['MLT'].to_numpy()

kalle = pd.read_pickle('/home/bing/Downloads/merged_substormlist.pd')

slt = np.array([datetime(t.year, t.month, t.day, t.hour, t.minute, t.second) for t in kalle.index])
slmlt = kalle['mlt']
slmlat = kalle['mlat']

f = (22 < slmlt) & (slmlt < 24) & (slmlat < 70) & (64 < slmlat) & (slt < datetime(2002, 1, 1)) & (datetime(2000, 1, 1) <= slt)

slt = slt[f]

#fig, axs = plt.subplots(1,2)
#axs[0].hist(slmlt, 100)
#axs[1].hist(slmlat, 100)

#%% NN good predictions

predicted_good = np.load('/home/bing/Dropbox/work/code/repos/icAnalyzer/data/predicted_good.npy')

#%% Fetch orbits available in all nc files

# Fetch all orbits
o = [int(o[-7:-3]) for o in sorted(glob.glob(p_in + '*.nc'))]

#%% Load all data

H, P, t = [], [], []
for i, orbit in tqdm(enumerate(o), total=len(o)):
    filename = p_in + f'or_{str(orbit).zfill(4)}.nc'
    cI = ConductanceImage(filename)

    '''
    if np.any(predicted_good[:, 0] == i):
        good_idx = predicted_good[predicted_good[:, 0] == i, 1]
        good_idx = np.sort(np.unique((good_idx[:, None] + np.arange(10)).flatten()))
            
        H.append(cI.H[good_idx])
        P.append(cI.P[good_idx])
        t.append(cI.time[good_idx])
    '''
    H.append(cI.H)
    P.append(cI.P)
    t.append(cI.time)
    
    
lat = cI.grid.lat
lt = (cI.grid.lon/15)%24
shape = cI.grid.shape

#%%

import os

for i, orbit in enumerate(o):
    for j, h in enumerate(H[i]):
        plt.ioff()
        plt.figure()
        plt.imshow(h)
        os.makedirs(f'/home/bing/Downloads/shit/{str(orbit).zfill(4)}', exist_ok=True)
        plt.savefig(f'/home/bing/Downloads/shit/{str(orbit).zfill(4)}/{j}.png')
        plt.close('all')
        plt.ion()

#%% Func

def get_sl_data(t, H, P, slt, os1, os2):
    os1 = timedelta(minutes=os1)
    os2 = timedelta(minutes=os2)

    Hs, Ps = [], []
    sc = 0
    for Hi, Pi, ti in zip(H, P, t):
        if np.any((ti[0] <= slt) & (slt <= ti[-1])):
            
            sl_id = np.argmin(abs(slt - ti[len(ti)//2]))
            slto = slt[sl_id]
            f = (slto + os1 <= ti) & (ti <= slto + os2)
            
            Hi, Pi = Hi[f], Pi[f]
            #mlt = slmlt[sl_id]
            #for j in range(Hi.shape[0]):
            #    Hi[j] = do_shift(Hi[j], mlt)
            #    Pi[j] = do_shift(Pi[j], mlt)
            
            Hs.append(Hi)
            Ps.append(Pi)
            sc += 1

    Hs  = np.concatenate(Hs, axis=0)
    Ps  = np.concatenate(Ps, axis=0)

    return Hs, Ps, sc

def do_shift(var, mlt):
    grid = cI.grid
    
    lon = grid.lon
    lon[lon<0] + 360
    
    lon_new = (lon - mlt*15)%360
    lon_new[lon_new > 180] -= 360
    xi_new, eta_new = grid.projection.geo2cube(lon_new, grid.lat)
    
    mask = ~np.isnan(var)
    var_new = griddata((xi_new[mask], eta_new[mask]), var[mask], (grid.xi, grid.eta), method='linear', fill_value=np.nan)
    
    return var_new

def do_PCA(Hs, Ps):
    # Assuming `data` is (N, 36, 36)
    H_f = Hs.reshape((Hs.shape[0], -1))  # (N, 1296)
    H_f[np.isnan(H_f)] = 0 # Imputing
    pcah = PCA(n_components=12)
    _ = pcah.fit_transform(H_f)

    P_f = Ps.reshape((Ps.shape[0], -1))  # (N, 1296)
    P_f[np.isnan(P_f)] = 0 # Imputing
    pcap = PCA(n_components=12)
    _ = pcap.fit_transform(P_f)
    
    return pcah, pcap

def plot_pca(pcah, pcap, title, fn):
    plt.ioff()
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    plt.subplots_adjust(hspace=0, wspace=0)
    for i, ax in enumerate(axs[:1, :].flatten()):
        pax = pp(ax)
        pax.plotimg(lat, lt, pcah.components_[i].reshape(shape))
        ax.axis('off')
        ax.text(.5, 1, f'PC {i+1}: {np.round(pcah.explained_variance_ratio_[i]*100,1)}\%', ha='center', va='center', fontsize=16, transform=ax.transAxes)
    for i, ax in enumerate(axs[1:, :].flatten()):
        pax = pp(ax)
        pax.plotimg(lat, lt, pcap.components_[i].reshape(shape))
        ax.axis('off')    
    axs[0,0].text(0, .5, 'Hall', ha='center', va='center', fontsize=20, transform=axs[0,0].transAxes, rotation='vertical')
    axs[1,0].text(0, .5, 'Pedersen', ha='center', va='center', fontsize=20, transform=axs[1,0].transAxes, rotation='vertical')
    for ax in axs[1, :]:
        ax.text(.5, 0, 'MLT 00', ha='center', va='center', fontsize=12, transform=ax.transAxes)

    plt.suptitle(title, fontsize=22)
    plt.savefig(fn, bbox_inches='tight')
    plt.close('all')
    plt.ion()

#%%

wsize = 6
for eh in range(0, 35, wsize):
    os1 = eh
    os2 = eh + wsize
    Hs, Ps, sc = get_sl_data(t, H, P, slt, os1, os2)
    
    f = np.array([np.sum(~np.isnan(hs))/(36**2) for hs in Hs]) > .9
    Hs = Hs[f]
    
    f = np.array([np.sum(~np.isnan(ps))/(36**2) for ps in Ps]) > .9
    Ps = Ps[f]
    
    print(sc)
    pcah, pcap = do_PCA(Hs, Ps)
    title = f'PCA: {sc} substorms: {Hs.shape[0]} images: {os1} - {os2} min'
    fn = f'/home/bing/Dropbox/work/code/repos/icAnalyzer/figures/naive_PCA_sl_pg_mlt_{os1}_{os2}.png'
    plot_pca(pcah, pcap, title, fn)

