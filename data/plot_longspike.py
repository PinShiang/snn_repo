import numpy as np
import matplotlib.pyplot as plt

filename = 'long0to9.npz'
spike_file = f'longSpikes/{filename}'


fig, axs = plt.subplots(1, 1, figsize=(20, 8))

try:
    # Load data
    print(spike_file)
    data = np.load(spike_file, allow_pickle=True)
    
    if 'arr_0' in data:
        time_data = data['arr_0'][0]
        unit_data = data['arr_0'][1]
    else:
        print(f"Warning")

    if len(time_data) > 0:
        max_time = np.max(time_data)
    else:
        max_time = 1


    axs.scatter(time_data, unit_data, s=0.5)
    axs.set_title(filename) 
    axs.set_xlim(0, max_time)
    axs.grid(True)
    axs.set_ylabel('Unit')
    axs.set_xlabel('Time')
        
except FileNotFoundError:
    axs.set_title(f"file Not Found")
    axs.text(0.5, 0.5, 'Missing', ha='center', va='center', transform=axs.transAxes)

plt.tight_layout()
plt.show()