import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 設定參數
max_time = 1.4  # 與你的訓練程式碼相同
nb_inputs = 700  # 輸入神經元數量
nb_steps = 100  # 時間步數
num_samples = 10  # 要繪製的樣本數量

# 載入測試資料
test_file = h5py.File("hdspikes/shd_test.h5", 'r')

x_test = test_file['spikes']
y_test = test_file['labels']

# 數字標籤名稱
digit_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

# 為每個數字類別找一個樣本
labels_array = np.array(y_test, dtype=int)
sample_indices = []

for digit in range(10):
    # 找到該數字的所有樣本索引
    digit_indices = np.where(labels_array == digit)[0]
    if len(digit_indices) > 1:
        # 選擇第一個樣本
        sample_indices.append(digit_indices[1])
    else:
        sample_indices.append(None)

# 建立時間區間用於 digitize
time_bins = np.linspace(0, max_time, num=nb_steps)

# 建立 2x5 的子圖
fig, axs = plt.subplots(2, 5, figsize=(20, 8))
axs = axs.ravel()

# 繪製每個樣本的 spike 圖
for i in range(10):
    idx = sample_indices[i]
    
    if idx is not None:
        # 取得該樣本的 spike 時間和單元
        firing_times = x_test['times'][idx]
        units_fired = x_test['units'][idx]
        
        # 使用 digitize 將連續時間轉換為離散時間步
        digitized_times = np.digitize(firing_times, time_bins)
        
        # 繪製 spike raster plot
        axs[i].scatter(digitized_times, units_fired, s=1, c='black', marker='|')
        axs[i].set_title(f'{digit_names[i]} (label={i})')
        axs[i].set_xlim(0, nb_steps)
        axs[i].set_ylim(0, nb_inputs)
        axs[i].grid(True, alpha=0.3)
        
        # 設定座標軸標籤
        if i % 5 == 0:
            axs[i].set_ylabel('Input Unit')
        if i >= 5:
            axs[i].set_xlabel('Time Step')
    else:
        axs[i].set_title(f"{digit_names[i]} (Not Found)")
        axs[i].text(0.5, 0.5, 'No data', ha='center', va='center', 
                   transform=axs[i].transAxes, fontsize=12)

plt.tight_layout()
plt.savefig('input_spike_patterns.png', dpi=150, bbox_inches='tight')
plt.show()

# 關閉檔案
test_file.close()

print("Input spike patterns saved to 'input_spike_patterns.png'")
print("\nSample statistics:")
for i in range(10):
    idx = sample_indices[i]
    if idx is not None:
        num_spikes = len(x_test['times'][idx])
        print(f"Digit {i} ({digit_names[i]}): {num_spikes} spikes")