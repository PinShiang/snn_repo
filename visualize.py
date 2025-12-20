
hdf5_file_path = "data/hdspikes/shd_test.h5"
import tables
import numpy as np
fileh = tables.open_file(hdf5_file_path, mode='r')
units = fileh.root.spikes.units
times = fileh.root.spikes.times
labels = fileh.root.extra.speaker
# labels = fileh.root.labels
# This is how we access spikes and labels
index = 0
print("Times (ms):", times[index])
print("Unit IDs:", units[index])
print("Label:", labels[index])
cnt = 0
for _ in labels:
  cnt = max(cnt, _)
print(f'cnt = {cnt}') 

# A quick raster plot for one of the samples
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(16,4))
idx = np.random.randint(len(times),size=3)
for i,k in enumerate(idx):
    ax = plt.subplot(1,3,i+1)
    ax.scatter(times[k],700-units[k], color="k", alpha=0.33, s=2)
    ax.set_title("Label %i"%labels[k])
    ax.axis("off")
print("Showing raster plots for 3 random samples")
plt.show()
print("Done")