import numpy as np
import matplotlib.pyplot as plt
import random
import glob
from tqdm import tqdm
from icreader import ConductanceImage

#%% Paths

p_in = '/home/bing/Dropbox/work/data/conductance/'

#%% Fetch orbits available in all nc files

# Fetch all orbits
o = [int(o[-7:-3]) for o in sorted(glob.glob(p_in + '*.nc'))]

#%%

H, P, dH, dP, w = [], [], [], [], []
for orbit in tqdm(o, total=len(o)):
    filename = p_in + f'or_{str(orbit).zfill(4)}.nc'
    cI = ConductanceImage(filename)
    H.append(cI.H)
    P.append(cI.P)
    dH.append(cI.dH)
    dP.append(cI.dP)
    w.append(cI.w)

#%%

start_from_safe_file = True

#%%

assert len(H) == len(dH) == len(P) == len(dP)
N = len(H)

seq_size = 5

if start_from_safe_file:
    q = np.load('/home/bing/Dropbox/work/temp_storage/seq_classification_results.npz')
    orbits = q['orbits']
    indices = q['indices']
    labels = q['labels']
    results = [(a,b,c) for (a,b,c) in zip(orbits, indices, labels)]
else:
    results = []

# -- Classification loop function --
def classify_images(H, P, dH, dP, w, classified = []):
    #classified = []
    keep_going = True

    if len(classified) != 0:
        cb = np.sum(np.array(results)[:, -1] == 0)
        cp = np.sum(np.array(results)[:, -1] == 1)
        cg = len(results) - cb - cp
    else:
        cb, cp, cg = 0, 0, 0
    while keep_going:
        ids = random.randint(0, N - 1)
        Hs, Ps, dHs, dPs, ws = H[ids], P[ids], dH[ids], dP[ids], w[ids]

        ns = Hs.shape[0]
        idt = random.randint(0, ns - 1)
        if (idt-seq_size) < 0:
            idt = seq_size
        if (idt + seq_size) > ns:
            idt = ns - seq_size
        start, stop = idt - seq_size, idt + seq_size
        
        idt = np.arange(start, stop+1)

        fig, axs = plt.subplots(5, 10, figsize=(20, 15))
        
        for ax, idi in zip(axs[0, :], idt):
            ax.imshow(Hs[idi], vmin=0, vmax=40)
            ax.set_title(f'Hall - Index {idi}')
        
        for ax, idi in zip(axs[1, :], idt):
            ax.imshow(dHs[idi], vmin=0, vmax=65)
            ax.set_title(f'Hall std - Index {idi}')
            
        for ax, idi in zip(axs[2, :], idt):
            ax.imshow(Ps[idi], vmin=0, vmax=10)
            ax.set_title(f'Pedersen - Index {idi}')
        
        for ax, idi in zip(axs[3, :], idt):
            ax.imshow(dPs[idi], vmin=0, vmax=65)
            ax.set_title(f'Pedersen std - Index {idi}')
        
        for ax, idi in zip(axs[4, :], idt):
            ax.imshow(ws[idi], vmin=0, vmax=1, cmap='Greys_r')
            ax.set_title(f'w - Index {idi}')
        
        for ax in axs.flatten():
            ax.axis('off')
            
        plt.suptitle(f"Press 1 (bad), 2 (partial), 3 (good), 5 (save and exit), #bad: {cb}, #partial: {cp}, #good: {cg}")
        plt.tight_layout()

        pressed = []

        def on_key(event):
            if event.key in ['1', '2', '3', '5']:
                pressed.append(event.key)
                plt.close()

        plt.savefig('/home/bing/Dropbox/work/temp_storage/seq_class.png', bbox_inches='tight')
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show(block=True)

        if pressed:
            key = pressed[0]
            if key == '1':
                classified.append((ids, idt[0], 0))
            elif key == '2':
                classified.append((ids, idt[0], 1))
            elif key == '3':
                classified.append((ids, idt[0], 2))
            elif key == '5':
                keep_going = False
                print("Stopping and saving...")

        cb = np.sum(np.array(classified)[:, -1] == 0)
        cp = np.sum(np.array(classified)[:, -1] == 1)
        cg = np.sum(np.array(classified)[:, -1] == 2)

    return classified

# -- Run the classification session --
results = classify_images(H, P, dH, dP, w, classified=results)

# -- Save the results --
if results:
    orbits, indices, labels = zip(*results)
    np.savez('/home/bing/Dropbox/work/temp_storage/seq_classification_results.npz', orbits = np.array(orbits), indices=np.array(indices), labels=np.array(labels))
    print(f"Saved {len(results)} classifications to 'classification_results.npz'")
else:
    print("No classifications made.")

#%%

q = np.load('/home/bing/Dropbox/work/temp_storage/seq_classification_results.npz')
orbits = q['orbits']
indices = q['indices']
labels = q['labels']


