import numpy as np
import matplotlib.pyplot as plt
import os

digit_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
#digit_names = [str(i) for i in range(10)]

fig, axs = plt.subplots(2, 5, figsize=(20, 8))
axs = axs.ravel()


for i, name in enumerate(digit_names):
    filename = f'{name}_v4' 
    spike_file = f'shiangNum/{filename}.npz'
    
    try:
        # Load data
        print(spike_file)
        data = np.load(spike_file, allow_pickle=True)
        
        if 'arr_0' in data:
            time_data = data['arr_0'][0]
            unit_data = data['arr_0'][1]
        else:
            print(f"Warning: {filename} ")
            continue

        if len(time_data) > 0:
            max_time = np.max(time_data)
        else:
            max_time = 1


        axs[i].scatter(time_data, unit_data, s=0.5)
        axs[i].set_title(filename) 
        axs[i].set_xlim(0, max_time)
        axs[i].grid(True)
        
        if i % 5 == 0:
            axs[i].set_ylabel('Unit')
        if i >= 5:
            axs[i].set_xlabel('Time')
            

    except FileNotFoundError:
        axs[i].set_title(f"{filename} (Not Found)")
        axs[i].text(0.5, 0.5, 'Missing', ha='center', va='center', transform=axs[i].transAxes)

plt.tight_layout()
plt.show()