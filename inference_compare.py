"""
Inference script for Spiking Neural Network (SNN) model.
Loads TWO trained models and performs predictions on input data to compare performance.
"""

import numpy as np
import torch
import torch.nn as nn
import os
import h5py
from pathlib import Path
import random
import re 

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
    
    # Clear CUDA cache to prevent OOM when loading multiple models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Loading model from: {filepath} ...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    
    try:
        if device.type == 'cuda':
            # torch.cuda.synchronize()
            w1 = checkpoint['w1'].to(device).requires_grad_(False) # Inference doesn't need grad
            w2 = checkpoint['w2'].to(device).requires_grad_(False)
            v1 = checkpoint['v1'].to(device).requires_grad_(False)
        else:
            w1 = checkpoint['w1'].to(device).requires_grad_(False)
            w2 = checkpoint['w2'].to(device).requires_grad_(False)
            v1 = checkpoint['v1'].to(device).requires_grad_(False)
    except RuntimeError as e:
        print(f"Error moving tensors to {device}: {e}")
        device = torch.device('cpu')
        w1 = checkpoint['w1'].to(device)
        w2 = checkpoint['w2'].to(device)
        v1 = checkpoint['v1'].to(device)
    
    result = {
        'w1': w1, 'w2': w2, 'v1': v1,
        'loss_hist': checkpoint.get('loss_hist', []),
        # Keep other params
        'nb_inputs': checkpoint.get('nb_inputs', 700),
        'nb_hidden': checkpoint.get('nb_hidden', 200),
        'nb_outputs': checkpoint.get('nb_outputs', 10),
        'nb_steps': checkpoint.get('nb_steps', 100),
        'max_time': checkpoint.get('max_time', 1.4),
        'time_step': checkpoint.get('time_step', 1e-3),
        'tau_mem': checkpoint.get('tau_mem', 10e-3),
        'tau_syn': checkpoint.get('tau_syn', 5e-3),
        'alpha': checkpoint.get('alpha', float(np.exp(-checkpoint.get('time_step', 1e-3)/checkpoint.get('tau_syn', 5e-3)))),
        'beta': checkpoint.get('beta', float(np.exp(-checkpoint.get('time_step', 1e-3)/checkpoint.get('tau_mem', 10e-3))))
    }
    
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
        
        mem = new_mem
        syn = new_syn
        
        spk_rec.append(out) # Only saving spikes to save memory if needed

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
    return out_rec, None # Simplified return

def predict(checkpoint, x_data, device=None, batch_size=None):
    w1 = checkpoint['w1']
    w2 = checkpoint['w2']
    v1 = checkpoint['v1']
    
    if device is None:
        device = w1.device
    
    # Get hyperparameters specifically for this model
    nb_inputs = checkpoint['nb_inputs']
    nb_hidden = checkpoint['nb_hidden']
    nb_outputs = checkpoint['nb_outputs']
    nb_steps = checkpoint['nb_steps']
    max_time = checkpoint['max_time']
    alpha = checkpoint['alpha']
    beta = checkpoint['beta']
    
    if batch_size is None:
        batch_size = 1
    
    dtype = torch.float

    try:
        # Preprocessing inputs
        # Note: If models have different nb_steps/max_time, this needs to be redone for each model
        x_processed = sparse_data_generator_from_dict(x_data, batch_size, nb_steps, nb_inputs, max_time, device)
        x_processed = x_processed.to_dense()
    except Exception as e:
        raise RuntimeError(f"Error processing input data: {e}") from e
    
    try:
        output, _ = run_snn(x_processed, w1, w2, v1, batch_size, nb_hidden, nb_outputs, nb_steps, alpha, beta, device, dtype)
        m, _ = torch.max(output, 1)
        _, am = torch.max(m, 1)
        predictions = am.detach().cpu().numpy()
        return predictions
    except RuntimeError as e:
        if 'CUDA' in str(e) or 'cuda' in str(e):
             # Fallback logic omitted for brevity in dual model run, assume CUDA works or CPU works
             raise e
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

import numpy as np
import torch
import os
import random
import re
import csv
import datetime
from pathlib import Path

# ... (前面的 class SurrGradSpike, load_model, predict 等函數保持不變，直接從上面複製即可) ...
# 為了節省篇幅，我這邊只列出有修改的 main 函數部分

def remove_ansi_codes(text):
    """移除 ANSI 顏色代碼，讓存入 txt 的文字變乾淨"""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ---------------------------------------------------------
    # 0. Setup Output Files (建立存檔機制)
    # ---------------------------------------------------------
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = "clean"
    csv_filename = f"comparison_results_{timestamp}.csv"
    txt_filename = f"comparison_report_{timestamp}.txt"
    
    print(f"Results will be saved to:\n 1. {csv_filename} (Excel data)\n 2. {txt_filename} (Readable report)")
    print("=" * 120)

    # 開啟檔案準備寫入
    csv_file = open(csv_filename, 'w', newline='', encoding='utf-8')
    txt_file = open(txt_filename, 'w', encoding='utf-8')
    
    csv_writer = csv.writer(csv_file)
    # 寫入 CSV Header
    csv_writer.writerow(['Filename', 'True_Label', 'Model_A_Pred', 'Model_A_Correct', 'Model_B_Pred', 'Model_B_Correct', 'Result_Type'])

    # 定義一個 helper function 同時輸出到螢幕和 txt 檔
    def log(msg, end='\n'):
        print(msg, end=end) # 印在螢幕 (有顏色)
        clean_msg = remove_ansi_codes(msg) # 移除顏色
        txt_file.write(clean_msg + end) # 寫入檔案

    # ---------------------------------------------------------
    # 1. Load Both Models
    # ---------------------------------------------------------
    model_path_1 = 'model/model_dropout0_testacc0.821.pth'
    model_path_2 = 'model/model_dropout0.1_testacc0.852.pth'
    
    m1_name = "Model A (Drop 0.0)"
    m2_name = "Model B (Drop 0.1)"

    try:
        ckpt1 = load_model(model_path_1, device=device)
        log(f"✓ Loaded {m1_name}")
        ckpt2 = load_model(model_path_2, device=device)
        log(f"✓ Loaded {m2_name}")
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        return

    log("=" * 120)
    
    # ---------------------------------------------------------
    # 2. Setup Data
    # ---------------------------------------------------------
    spikes_dir = 'data/SNN/spikes/clean_spikes'
    
    if not os.path.exists(spikes_dir):
        print(f"Error: Directory '{spikes_dir}' does not exist!")
        return
    
    target_count = 1000 # 您圖片中是 500
    spike_files = []
    iterator = Path(spikes_dir).glob("*.npz")

    # Random Sampling
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
    
    log(f"\nFound {len(spike_files)} spike file(s) for comparison.")
    # 調整欄位寬度: Filename 改為 60，確保對齊
    header = f"{'Filename':<60} | {'True':<5} | {m1_name:<20} | {m2_name:<20} | {'Result':<15}"
    log(header)
    log("-" * 120)
    
    # ---------------------------------------------------------
    # 3. Inference & Comparison Loop
    # ---------------------------------------------------------
    
    stats = {
        'm1_correct': 0,
        'm2_correct': 0,
        'both_correct': 0,
        'both_wrong': 0,
        'm1_only_correct': 0,
        'm2_only_correct': 0,
        'total': 0
    }
    
    error_log = [] 
    label_pattern = re.compile(r'digit-(\d+)')

    for i, spike_file in enumerate(spike_files, 1):
        filename = spike_file.name
        
        match = label_pattern.search(filename)
        true_label = int(match.group(1)) if match else -1
        
        try:
            input_data = load_spike_file(str(spike_file))
            
            # Predict
            pred1_arr = predict(ckpt1, input_data, device=device)
            p1 = int(pred1_arr[0])
            
            pred2_arr = predict(ckpt2, input_data, device=device)
            p2 = int(pred2_arr[0])
            
            correct1 = (p1 == true_label)
            correct2 = (p2 == true_label)
            
            if correct1: stats['m1_correct'] += 1
            if correct2: stats['m2_correct'] += 1
            stats['total'] += 1

            # Determine Result String & Color
            result_str_clean = ""
            if correct1 and correct2:
                stats['both_correct'] += 1
                result_icon = "✓ Both"
                result_str_clean = "Both Correct"
                color_start = "\033[92m" # Green
            elif not correct1 and not correct2:
                stats['both_wrong'] += 1
                result_icon = "✗ Both"
                result_str_clean = "Both Wrong"
                color_start = "\033[91m" # Red
            elif correct1 and not correct2:
                stats['m1_only_correct'] += 1
                result_icon = "← M1 Win"
                result_str_clean = "M1 Win"
                color_start = "\033[93m" # Yellow
            elif not correct1 and correct2:
                stats['m2_only_correct'] += 1
                result_icon = "→ M2 Win"
                result_str_clean = "M2 Win"
                color_start = "\033[94m" # Blue
            
            color_end = "\033[0m"

            # 準備顯示用的字串
            p1_disp = f"{p1} {'(✓)' if correct1 else '(✗)'}"
            p2_disp = f"{p2} {'(✓)' if correct2 else '(✗)'}"
            
            # 格式化輸出行 (Filename 寬度設為 60)
            row_str = f"{filename:<60} | {true_label:<5} | {p1_disp:<20} | {p2_disp:<20} | {color_start}{result_icon:<15}{color_end}"
            
            # 使用 log 函式輸出
            log(row_str)

            # 寫入 CSV
            csv_writer.writerow([filename, true_label, p1, correct1, p2, correct2, result_str_clean])

            if not correct1 or not correct2:
                error_log.append({
                    'file': filename, 'true': true_label, 
                    'p1': p1, 'p1_ok': correct1,
                    'p2': p2, 'p2_ok': correct2
                })

        except Exception as e:
            log(f"{filename:<60} | ERROR: {e}")

    # ---------------------------------------------------------
    # 4. Final Summary
    # ---------------------------------------------------------
    total = stats['total']
    if total == 0: 
        csv_file.close()
        txt_file.close()
        return

    acc1 = (stats['m1_correct'] / total) * 100
    acc2 = (stats['m2_correct'] / total) * 100

    log("\n" + "=" * 120)
    log("COMPARISON SUMMARY")
    log("=" * 120)
    log(f"Total Samples: {total}")
    log(f"{m1_name} Accuracy: {acc1:.2f}% ({stats['m1_correct']}/{total})")
    log(f"{m2_name} Accuracy: {acc2:.2f}% ({stats['m2_correct']}/{total})")
    
    log("-" * 50)
    log("Difference Matrix:")
    log(f"Both Correct:      {stats['both_correct']:<5} (Easy cases)")
    log(f"Both Wrong:        {stats['both_wrong']:<5} (Hard cases)")
    log(f"Only M1 Correct:   {stats['m1_only_correct']:<5} (M1 performed better here)")
    log(f"Only M2 Correct:   {stats['m2_only_correct']:<5} (M2 performed better here)")
    log("-" * 50)
    
    if stats['m2_only_correct'] > stats['m1_only_correct']:
        diff = stats['m2_only_correct'] - stats['m1_only_correct']
        log(f"CONCLUSION: {m2_name} is better by {diff} sample(s).")
    elif stats['m1_only_correct'] > stats['m2_only_correct']:
        diff = stats['m1_only_correct'] - stats['m2_only_correct']
        log(f"CONCLUSION: {m1_name} is better by {diff} sample(s).")
    else:
        log("CONCLUSION: Both models have identical net performance.")
        
    log("=" * 120)

    if stats['m1_only_correct'] > 0 or stats['m2_only_correct'] > 0:
        log("\nDisagreement Details (Where models differed):")
        log(f"{'Filename':<60} | True | M1 | M2")
        for err in error_log:
            if err['p1_ok'] != err['p2_ok']:
                winner = "M1" if err['p1_ok'] else "M2"
                log(f"{err['file']:<60} |  {err['true']}   | {err['p1']}  | {err['p2']}  <-- {winner} correct")

    # 關閉檔案
    csv_file.close()
    txt_file.close()
    print(f"\n[Done] Data saved to {csv_filename} and {txt_filename}")

if __name__ == "__main__":
    main()