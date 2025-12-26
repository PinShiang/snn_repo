"""
Inference script for Spiking Neural Network (SNN) model.
Loads a trained model and performs predictions on input data.
"""

import numpy as np
import torch
import torch.nn as nn
import os
import h5py
from pathlib import Path
import random
import re  # 新增：用於解析檔名中的 label
# from utils import get_shd_dataset # 如果沒有用到可以註解掉

# -------------------------------------------------------------------------
# Spiking Functions & Model Loading (保持不變)
# -------------------------------------------------------------------------

class SurrGradSpike(torch.autograd.Function):
    scale = 100.0 

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
        return grad

spike_fn = SurrGradSpike.apply

def load_model(filepath, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize()
        except RuntimeError as e:
            print(f"Warning: Clearing CUDA error: {e}")
            torch.cuda.empty_cache()
    
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    
    try:
        if device.type == 'cuda':
            torch.cuda.synchronize()
            w1 = checkpoint['w1'].to(device).requires_grad_(True)
            torch.cuda.synchronize()
            w2 = checkpoint['w2'].to(device).requires_grad_(True)
            torch.cuda.synchronize()
            v1 = checkpoint['v1'].to(device).requires_grad_(True)
            torch.cuda.synchronize()
        else:
            w1 = checkpoint['w1'].to(device).requires_grad_(True)
            w2 = checkpoint['w2'].to(device).requires_grad_(True)
            v1 = checkpoint['v1'].to(device).requires_grad_(True)
    except RuntimeError as e:
        print(f"Error moving tensors to {device}: {e}")
        device = torch.device('cpu')
        w1 = checkpoint['w1'].to(device).requires_grad_(True)
        w2 = checkpoint['w2'].to(device).requires_grad_(True)
        v1 = checkpoint['v1'].to(device).requires_grad_(True)
    
    print(f"Model loaded from {filepath}")
    
    result = {
        'w1': w1, 'w2': w2, 'v1': v1,
        'loss_hist': checkpoint.get('loss_hist', [])
    }
    for key, value in checkpoint.items():
        if key not in ['w1', 'w2', 'v1', 'loss_hist']:
            result[key] = value
    
    return result

def sparse_data_generator_from_dict(X, batch_size, nb_steps, nb_units, max_time, device, shuffle=True):
    if not isinstance(X, dict):
        raise ValueError(f"Expected X to be a dictionary, got {type(X)}")
    
    firing_times = np.asarray(X['times'], dtype=np.float32)
    units_fired = np.asarray(X['units'], dtype=np.int64)
    
    if batch_size != 1:
        batch_size = 1
    
    valid_mask = firing_times < max_time
    firing_times = firing_times[valid_mask]
    units_fired = units_fired[valid_mask]
    
    time_bins = np.linspace(0, max_time, num=nb_steps)
    times = np.digitize(firing_times, time_bins)
    
    if np.any(units_fired < 0) or np.any(units_fired >= nb_units):
        units_fired = np.clip(units_fired, 0, nb_units - 1)
    
    units = units_fired
    batch = [0 for _ in range(len(times))]
    
    coo = [ [] for i in range(3) ]
    coo[0].extend(batch)
    coo[1].extend(times)
    coo[2].extend(units)

    i = torch.LongTensor(coo).to(device)
    v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)

    X_batch = torch.sparse_coo_tensor(i, v, torch.Size([batch_size, nb_steps, nb_units]), device=device)
    return X_batch.to(device=device)

def run_snn(inputs, w1, w2, v1, batch_size, nb_hidden, nb_outputs, nb_steps, alpha, beta, device, dtype):
    syn = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
    mem = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)

    mem_rec = []
    spk_rec = []

    out = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
    h1_from_input = torch.einsum("abc,cd->abd", (inputs, w1))
    for t in range(nb_steps):
        h1 = h1_from_input[:, t] + torch.einsum("ab,bc->ac", (out, v1))
        mthr = mem - 1.0
        out = spike_fn(mthr)
        rst = out.detach()

        new_syn = alpha * syn + h1
        new_mem = (beta * mem + syn) * (1.0 - rst)

        mem_rec.append(mem)
        spk_rec.append(out)
        
        mem = new_mem
        syn = new_syn

    mem_rec = torch.stack(mem_rec, dim=1)
    spk_rec = torch.stack(spk_rec, dim=1)

    h2 = torch.einsum("abc,cd->abd", (spk_rec, w2))
    flt = torch.zeros((batch_size, nb_outputs), device=device, dtype=dtype)
    out = torch.zeros((batch_size, nb_outputs), device=device, dtype=dtype)
    out_rec = [out]
    for t in range(nb_steps):
        new_flt = alpha * flt + h2[:, t]
        new_out = beta * out + flt

        flt = new_flt
        out = new_out

        out_rec.append(out)

    out_rec = torch.stack(out_rec, dim=1)
    other_recs = [mem_rec, spk_rec]
    return out_rec, other_recs

def predict(checkpoint, x_data, y_data=None, device=None, batch_size=None):
    w1 = checkpoint['w1']
    w2 = checkpoint['w2']
    v1 = checkpoint['v1']
    
    if device is None:
        device = w1.device
    
    nb_inputs = checkpoint.get('nb_inputs', 700)
    nb_hidden = checkpoint.get('nb_hidden', 200)
    nb_outputs = checkpoint.get('nb_outputs', 10)
    nb_steps = checkpoint.get('nb_steps', 100)
    max_time = checkpoint.get('max_time', 1.4)
    
    if batch_size is None:
        batch_size = 1
        
    time_step = checkpoint.get('time_step', 1e-3)
    tau_mem = checkpoint.get('tau_mem', 10e-3)
    tau_syn = checkpoint.get('tau_syn', 5e-3)
    alpha = checkpoint.get('alpha', float(np.exp(-time_step/tau_syn)))
    beta = checkpoint.get('beta', float(np.exp(-time_step/tau_mem)))
    
    dtype = torch.float
    
    # Optional: Comment out print to make output cleaner
    # print(f'batch_size: {batch_size}, nb_steps: {nb_steps}, nb_inputs: {nb_inputs}, max_time: {max_time}')

    try:
        x_data = sparse_data_generator_from_dict(x_data, batch_size, nb_steps, nb_inputs, max_time, device)
        x_data = x_data.to_dense()
    except Exception as e:
        raise RuntimeError(f"Error processing input data: {e}") from e
    
    try:
        output, _ = run_snn(x_data, w1, w2, v1, batch_size, nb_hidden, nb_outputs, nb_steps, alpha, beta, device, dtype)
        m, _ = torch.max(output, 1)
        _, am = torch.max(m, 1)
        predictions = am.detach().cpu().numpy()
        return predictions
    except RuntimeError as e:
        # Fallback to CPU if CUDA fails (simplified for brevity)
        if 'CUDA' in str(e) or 'cuda' in str(e):
            device = torch.device('cpu')
            w1 = w1.to(device); w2 = w2.to(device); v1 = v1.to(device)
            x_data = x_data.to(device)
            output, _ = run_snn(x_data, w1, w2, v1, batch_size, nb_hidden, nb_outputs, nb_steps, alpha, beta, device, dtype)
            m, _ = torch.max(output, 1)
            _, am = torch.max(m, 1)
            predictions = am.detach().cpu().numpy()
            return predictions
        else:
            raise

def load_spike_file(filepath):
    my_data = np.load(filepath, allow_pickle=True)
    if 'arr_0' in my_data:
        data_array = my_data['arr_0']
        data_times = data_array[0]
        data_units = data_array[1]
    elif 'times' in my_data and 'units' in my_data:
        data_times = my_data['times']
        data_units = my_data['units']
    else:
        raise ValueError(f"Unknown data format.")
    
    return {'times': np.asarray(data_times, dtype=np.float32), 'units': np.asarray(data_units, dtype=np.int64)}


# -------------------------------------------------------------------------
# Main Execution (已修改)
# -------------------------------------------------------------------------

def main():
    """Main execution function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("=" * 70)
    
    # Load the model
    # 請確認路徑是否正確
    checkpoint1 = load_model('model/model_dropout0_testacc0.821.pth', device=device)
    
    checkpoint2 = load_model('model/model_dropout0.1_testacc0.852.pth', device=device)
    print("=" * 70)
    
    spikes_dir = 'data/SNN/spikes/noisy_spikes'
    
    if not os.path.exists(spikes_dir):
        print(f"Error: Directory '{spikes_dir}' does not exist!")
        return
    
    # 設定測試數量與檔案選取邏輯
    target_count = 100
    spike_files = []
    iterator = Path(spikes_dir).glob("*.npz")

    for i, file_path in enumerate(iterator):
        if i < target_count:
            spike_files.append(file_path)
        else:
            j = random.randint(0, i)
            if j < target_count:
                spike_files[j] = file_path
    
    if not spike_files:
        print(f"No .npz files found in '{spikes_dir}'")
        return
    
    print(f"\nFound {len(spike_files)} spike file(s) in {spikes_dir} (randomly sampled)")
    print("=" * 70)
    
    results = []
    error_log = [] # 用來記錄錯誤的詳細資訊
    
    correct_count = 0
    total_processed = 0
    
    # Regex pattern to extract digit label (e.g., from '...digit-5.npz')
    # 尋找 "digit-" 後面跟著的數字
    label_pattern = re.compile(r'digit-(\d+)')

    for i, spike_file in enumerate(spike_files, 1):
        filename = spike_file.name
        print(f"\n[{i}/{len(spike_files)}] Processing: {filename}")
        
        # 1. 嘗試從檔名解析正確 Label
        match = label_pattern.search(filename)
        true_label = -1
        if match:
            true_label = int(match.group(1))
            print(f"   -> True Label extracted: {true_label}")
        else:
            print(f"   -> Warning: Could not extract label from filename.")

        try:
            # Load data
            input_data = load_spike_file(str(spike_file))
            
            # Prediction
            prediction = predict(checkpoint, input_data, device=device, batch_size=1)
            predicted_class = int(prediction[0])
            
            # 2. 判斷對錯
            is_correct = False
            status_str = "Unknown"
            
            if true_label != -1:
                if predicted_class == true_label:
                    is_correct = True
                    status_str = "CORRECT"
                    correct_count += 1
                else:
                    is_correct = False
                    status_str = "WRONG"
                    # 記錄錯誤資訊
                    error_log.append({
                        'filename': filename,
                        'true': true_label,
                        'pred': predicted_class
                    })
            
            total_processed += 1
            
            results.append({
                'filename': filename,
                'prediction': predicted_class,
                'true_label': true_label,
                'is_correct': is_correct
            })
            
            # 簡潔的輸出結果
            if is_correct:
                print(f"   -> Prediction: {predicted_class} | Status: \033[92m{status_str}\033[0m") # 綠色文字
            else:
                print(f"   -> Prediction: {predicted_class} | Status: \033[91m{status_str}\033[0m (Expected: {true_label})") # 紅色文字
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("INFERENCE SUMMARY")
    print("=" * 70)
    print(f"Total files processed: {total_processed}")
    
    if total_processed > 0:
        accuracy = (correct_count / total_processed) * 100
        print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{total_processed})")
    else:
        print("Accuracy: N/A")

    print("-" * 70)
    
    # 3. 顯示錯誤報告
    if error_log:
        print(f"Missclassified Items ({len(error_log)}):")
        print(f"{'Filename':<40} {'True':<10} {'Predicted':<10}")
        print("-" * 70)
        for err in error_log:
            print(f"{err['filename']:<40} {err['true']:<10} {err['pred']:<10}")
    else:
        print("Perfect! No misclassifications found.")
        
    print("=" * 70)

if __name__ == "__main__":
    main()