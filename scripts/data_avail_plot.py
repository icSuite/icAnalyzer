# --- 4.1 Load Data ---
gos = pd.read_csv(PATHS['segments'], header=None)
gos.columns = ['orbit', 'shit_start', 'shit_stop', 'good_start', 'good_stop']

t = []
nans = []
for orbit in tqdm(gos['orbit'], total=gos.shape[0]):
    cI = ConductanceImage(pjoin(PATHS['data_in'], f'or_{str(orbit).zfill(4)}.nc'))
    t.append(cI.time)
    nans.append(np.sum(np.isnan(cI.H) | np.isnan(cI.P), axis=(1,2)) / (36*36) * 100)

#%%

dates = np.concatenate(t)
data_avail = 100 - np.concatenate(nans)

# Convert to days
dates = dates.astype("datetime64[M]")
dates_75 = dates[data_avail >= 75]
dates_50 = dates[data_avail >= 50]
dates_25 = dates[data_avail >= 25]

plt.figure(figsize=(15, 9))

unique, counts = np.unique(dates, return_counts=True)
plt.bar(unique.astype(str), counts, label=f'Total: {int(np.sum(counts))}')

unique, counts = np.unique(dates_25, return_counts=True)
plt.bar(unique.astype(str), counts, label=f'Min 25%: {int(np.sum(counts))}')

unique, counts = np.unique(dates_50, return_counts=True)
plt.bar(unique.astype(str), counts, label=f'Min 50%: {int(np.sum(counts))}')

unique, counts = np.unique(dates_75, return_counts=True)
plt.bar(unique.astype(str), counts, label=f'Min 75%: {int(np.sum(counts))}')

plt.grid(axis='y')
plt.xlabel("Date", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.xticks(rotation=45, fontsize=16)
plt.gca().set_xticks(plt.gca().get_xticks()[::2])
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.title('IMAGE: available data', fontsize=24)
plt.show()
