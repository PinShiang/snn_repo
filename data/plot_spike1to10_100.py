import numpy as np
import matplotlib.pyplot as plt

digit_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
#digit_names = [str(i) for i in range(10)]

# Parameters
nb_steps = 100  # x axis (time axis)
nb_units = 700  # y axis (neuron axis)
max_time = 1.4  # zero padding to 1.4 seconds

fig, axs = plt.subplots(2, 5, figsize=(20, 8))
axs = axs.ravel()

for i, name in enumerate(digit_names):
    if i==0:
        continue
    filename = f'{name}_v1' 
    spike_file = f'myOneToTen/{filename}.npz'
    
    try:
        # Load data
        print(f"Processing: {spike_file}")
        data = np.load(spike_file, allow_pickle=True)
        
        if 'arr_0' in data:
            time_data = data['arr_0'][0]
            unit_data = data['arr_0'][1]
        else:
            print(f"Warning: {filename} - arr_0 not found")
            continue
        
        # Create time bins from 0 to 1.4s (zero padding)
        time_bins = np.linspace(0, max_time, num=nb_steps)
        
        # Filter out spikes beyond 1.4s (if any)
        valid_mask = time_data < max_time
        time_data = time_data[valid_mask]
        unit_data = unit_data[valid_mask]
        
        # Discretize time into corresponding bins
        time_indices = np.digitize(time_data, time_bins)
        time_indices = np.clip(time_indices, 0, nb_steps)
        
        # Ensure unit_data is within valid range
        unit_indices = np.clip(unit_data.astype(int), 0, nb_units - 1)
        
        # Create sparse matrix using numpy (x-axis=time, y-axis=neurons)
        sparse_matrix = np.zeros((nb_units, nb_steps), dtype=np.int8)
        
        # Set spikes to 1
        sparse_matrix[unit_indices, time_indices] = 1
        
        # Visualize
        axs[i].imshow(sparse_matrix, aspect='auto', cmap='binary', 
                      interpolation='nearest', origin='lower')
        axs[i].set_title(f"{filename}\n({len(time_data)} spikes, 0-1.4s)")
        axs[i].set_xlabel('Time bins (0-1.4s / 99 bins)')
        axs[i].set_ylabel('Neuron units (0-700)')
        
        # Display sparsity info
        sparsity = 100 * (1 - np.count_nonzero(sparse_matrix) / sparse_matrix.size)
        #axs[i].text(0.02, 0.98, f'Sparsity: {sparsity:.1f}%', 
        #            transform=axs[i].transAxes, 
        #            verticalalignment='top',
        #            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        print(f"  - Shape: {sparse_matrix.shape}")
        print(f"  - Non-zero elements: {np.count_nonzero(sparse_matrix)}")
        print(f"  - Sparsity: {sparsity:.2f}%\n")
            
    except FileNotFoundError:
        axs[i].set_title(f"{filename} (Not Found)")
        axs[i].text(0.5, 0.5, 'File Missing', 
                    ha='center', va='center', 
                    transform=axs[i].transAxes,
                    fontsize=14, color='red')
        axs[i].axis('off')
    except Exception as e:
        axs[i].set_title(f"{filename} (Error)")
        axs[i].text(0.5, 0.5, f'Error:\n{str(e)}', 
                    ha='center', va='center', 
                    transform=axs[i].transAxes,
                    fontsize=10, color='red')
        axs[i].axis('off')

plt.tight_layout()
plt.savefig('sparse_matrix_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print("Visualization complete!")