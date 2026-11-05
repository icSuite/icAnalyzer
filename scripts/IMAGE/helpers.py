#%% Import

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
import glob
from icreader import ConductanceImage, load


#%% Functions

def load_data(PATHS, spar=1):
    
    # --- 4.1 Load and Process Data ---
    gos = pd.read_csv(PATHS['segments'], header=None)
    gos.columns = ['orbit', 'shit_start', 'shit_stop', 'good_start', 'good_stop']
    gos = gos.loc[gos['orbit'] < 503, :]

    t, H, P, W, o = [], [], [], [], []
    grid = None
    for orbit in tqdm(gos['orbit'], total=gos.shape[0]):
        try:
            cI = ConductanceImage(pjoin(PATHS['data_in'], f'or_{str(orbit).zfill(4)}.nc'))
            start = gos.loc[gos['orbit']==orbit].iloc[0]['good_start']
            stop = gos.loc[gos['orbit']==orbit].iloc[0]['good_stop']
            
            #if stop == 0 or np.isnan(start): continue
            if stop == 0: continue # New
            if np.isnan(start): start = 0 # New
            if np.isnan(stop): stop = cI.H.shape[0]-1
            
            start, stop = int(start), int(stop+1)
            H.append(cI.H[start:stop])
            P.append(cI.P[start:stop])
            W.append(cI.w[start:stop])
            t.append(cI.time[start:stop])
            o.append(np.ones(cI.time[start:stop].size)*orbit)
            if grid is None:
                grid = cI.grid
        except Exception as e:
            continue

    H = np.concatenate(H, axis=0)
    P = np.concatenate(P, axis=0)
    W = np.concatenate(W, axis=0)
    t = np.concatenate(t)
    o = np.concatenate(o)
    
    f = np.sum(np.isnan(H) | np.isnan(P) | np.isnan(W), axis=(1,2)) / (H.shape[1] * H.shape[2])
    f = f <= spar

    return H[f], P[f], W[f], t[f], o[f], grid

def load_data_new(PATHS, spar=1):
    
    # --- 4.1 Load and Process Data ---
    gos = pd.read_csv(PATHS['segments'], header=None)
    gos.columns = ['orbit', 'shit_start', 'shit_stop', 'good_start', 'good_stop']
    gos = gos.loc[gos['orbit'] < 503, :]

    t, H, P, coverage, W, o = [], [], [], [], [], []
    grid = None
    for orbit in tqdm(gos['orbit'], total=gos.shape[0]):
        try:
            cI = load(pjoin(PATHS['data_in'], f'or_{str(orbit).zfill(4)}.nc'))
            start = gos.loc[gos['orbit']==orbit].iloc[0]['good_start']
            stop = gos.loc[gos['orbit']==orbit].iloc[0]['good_stop']
            
            #if stop == 0 or np.isnan(start): continue
            if stop == 0: continue # New
            if np.isnan(start): start = 0 # New
            if np.isnan(stop): stop = cI.H.shape[0]-1
            
            start, stop = int(start), int(stop+1)
            H.append(cI.H[start:stop])
            P.append(cI.P[start:stop])
            coverage.append(cI.coverage[start:stop])
            W.append(cI.method_quality_weight[start:stop])
            t.append(cI.time[start:stop])
            o.append(np.ones(cI.time[start:stop].size)*orbit)
            if grid is None:
                grid = cI.grid
        except Exception as e:
            continue

    H = np.concatenate(H, axis=0)
    P = np.concatenate(P, axis=0)
    W = np.concatenate(W, axis=0)
    coverage = np.concatenate(coverage, axis=0)
    t = np.concatenate(t)
    o = np.concatenate(o)
    
    H[coverage < .8] = np.nan
    P[coverage < .8] = np.nan
    W[coverage < .8] = np.nan
    
    f = np.sum(np.isnan(H) | np.isnan(P) | np.isnan(W), axis=(1,2)) / (H.shape[1] * H.shape[2])
    f = f <= spar

    return H[f], P[f], W[f], coverage[f], t[f], o[f], grid

def load_data_wic(PATHS, spar=1):
    
    # --- 4.1 Load and Process Data ---
    gos = pd.read_csv(PATHS['segments'], header=None)
    gos.columns = ['orbit', 'shit_start', 'shit_stop', 'good_start', 'good_stop']
    gos = gos.loc[gos['orbit'] < 503, :]

    t, wic, wic_std, s12, s12_std, W, o = [], [], [], [], [], [], []
    grid = None
    for orbit in tqdm(gos['orbit'], total=gos.shape[0]):
        try:
            cI = ConductanceImage(pjoin(PATHS['data_in'], f'or_{str(orbit).zfill(4)}.nc'))
            start = gos.loc[gos['orbit']==orbit].iloc[0]['good_start']
            stop = gos.loc[gos['orbit']==orbit].iloc[0]['good_stop']
            
            #if stop == 0 or np.isnan(start): continue
            if stop == 0: continue # New
            if np.isnan(start): start = 0 # New
            if np.isnan(stop): stop = cI.H.shape[0]-1
            
            start, stop = int(start), int(stop+1)
            wic.append(cI.wic_avg[start:stop])
            s12.append(cI.s12_avg[start:stop])
            wic_std.append(cI.wic_std[start:stop])
            s12_std.append(cI.s12_std[start:stop])
            W.append(cI.w[start:stop])
            t.append(cI.time[start:stop])
            o.append(np.ones(cI.time[start:stop].size)*orbit)
            if grid is None:
                grid = cI.grid
        except Exception as e:
            continue
    wic = np.concatenate(wic, axis=0)
    wic_std = np.concatenate(wic_std, axis=0)
    s12 = np.concatenate(s12, axis=0)
    s12_std = np.concatenate(s12_std, axis=0)
    W = np.concatenate(W, axis=0)
    t = np.concatenate(t)
    o = np.concatenate(o)
    
    f = np.sum(np.isnan(wic) | np.isnan(wic_std) | np.isnan(s12) | np.isnan(s12_std) | np.isnan(W), axis=(1,2)) / (wic.shape[1] * wic.shape[2])
    f = f <= spar

    return wic[f], wic_std[f], s12[f], s12_std[f], W[f], t[f], o[f], grid

def load_data_wic_new(PATHS, spar=1):

    # --- 4.1 Load and Process Data ---
    gos = pd.read_csv(PATHS['segments'], header=None)
    gos.columns = ['orbit', 'shit_start', 'shit_stop', 'good_start', 'good_stop']
    gos = gos.loc[gos['orbit'] < 503, :]

    t, wic, wic_std, s12, s12_std, W, o = [], [], [], [], [], [], []
    grid = None
    for orbit in tqdm(gos['orbit'], total=gos.shape[0]):
        try:
            file = pjoin(PATHS['data_in'], f'or_{str(orbit).zfill(4)}.nc')
            with load(file) as cI:
                start = gos.loc[gos['orbit']==orbit].iloc[0]['good_start']
                stop = gos.loc[gos['orbit']==orbit].iloc[0]['good_stop']

                if stop == 0:
                    continue

                source_indices = np.asarray(cI.source_indices['wic'])
                if np.isnan(start):
                    start = 0
                if np.isnan(stop):
                    stop = np.nanmax(source_indices)

                start, stop = int(start), int(stop)
                selection = np.flatnonzero(
                    (source_indices >= start) & (source_indices <= stop)
                )
                if selection.size == 0:
                    continue

                wic.append(np.asarray(cI.wic_corrected[selection]))
                s12.append(np.asarray(cI.si12[selection]))
                wic_std.append(np.asarray(cI.dwic_corrected[selection]))
                s12_std.append(np.asarray(cI.dsi12[selection]))
                W.append(np.asarray(cI.method_quality_weight[selection]))
                selected_time = np.asarray(cI.time[selection])
                t.append(selected_time)
                o.append(np.full(selected_time.size, orbit))
                if grid is None:
                    grid = cI.grid
        except Exception as e:
            continue
    wic = np.concatenate(wic, axis=0)
    wic_std = np.concatenate(wic_std, axis=0)
    s12 = np.concatenate(s12, axis=0)
    s12_std = np.concatenate(s12_std, axis=0)
    W = np.concatenate(W, axis=0)
    t = np.concatenate(t)
    o = np.concatenate(o)

    f = np.sum(np.isnan(wic) | np.isnan(wic_std) | np.isnan(s12) | np.isnan(s12_std) | np.isnan(W), axis=(1,2)) / (wic.shape[1] * wic.shape[2])
    f = f <= spar

    return wic[f], wic_std[f], s12[f], s12_std[f], W[f], t[f], o[f], grid


def load_all_data(PATHS):
    
    # --- 4.1 Load and Process Data ---
    gos = pd.read_csv(PATHS['segments'], header=None)
    gos.columns = ['orbit', 'shit_start', 'shit_stop', 'good_start', 'good_stop']
    #gos = gos.loc[gos['orbit'] < 503, :]

    t, H, P, W, o = [], [], [], [], []
    for orbit in tqdm(gos['orbit'], total=gos.shape[0]):
        try:
            cI = ConductanceImage(pjoin(PATHS['data_in'], f'or_{str(orbit).zfill(4)}.nc'))
            H.append(cI.H)
            P.append(cI.P)
            W.append(cI.w)
            t.append(cI.time)
            o.append(np.ones(cI.time.size)*orbit)
        except Exception as e:
            continue

    H = np.concatenate(H, axis=0)
    P = np.concatenate(P, axis=0)
    W = np.concatenate(W, axis=0)
    t = np.concatenate(t)
    o = np.concatenate(o)

    return H, P, W, t, o
