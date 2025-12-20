"""
Combined Inference Script: SNN-VAD + Digit Classification
Performs segmentation on a long audio spike file and predicts the digit sequence.
"""

import numpy as np
import torch
import os
from pathlib import Path

# --- 1. SNN Components (Activation & Model Loading) ---

class SurrGradSpike(torch.autograd.Function):
    """
    Spiking nonlinearity which also implements the surrogate gradient.
    """
    scale = 100.0  # controls steepness of surrogate gradient

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

# Spike function
spike_fn = SurrGradSpike.apply

def load_model(filepath, device=None):
    """Load a saved model checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    
    try:
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

# --- 2. Data Processing & VAD ---

def snn_vad_segmentation(full_times, full_units, 
                         window_size=0.05,    # 50ms window
                         threshold_start=70,  # Threshold to start segment
                         threshold_end=20,    # Threshold to end segment
                         min_duration=0.05,   # Discard segments shorter than this
                         padding=0.1,         # Buffer time
                         target_length=1.4):  # Target fixed length for output
    """
    Perform Voice Activity Detection (VAD) on spike data.
    Returns a list of segmented spike clips, normalized to a fixed target length.
    """
    
    # 1. Ensure data is sorted
    sort_idx = np.argsort(full_times)
    full_times = full_times[sort_idx]
    full_units = full_units[sort_idx]
    
    max_time = full_times.max() if len(full_times) > 0 else 0
    if max_time == 0:
        return []

    # 2. Histogram for sliding window
    bins = np.arange(0, max_time + window_size, window_size)
    counts, _ = np.histogram(full_times, bins=bins)
    
    segments = []
    is_active = False
    start_time = 0
    silence_counter = 0
    max_silence_bins = int(0.2 / window_size) 
    
    # 3. State Machine Scan
    for i, count in enumerate(counts):
        current_time = i * window_size
        
        if not is_active:
            if count > threshold_start:
                is_active = True
                start_time = max(0, current_time - padding)
                silence_counter = 0
        else:
            if count < threshold_end:
                silence_counter += 1
                if silence_counter > max_silence_bins:
                    is_active = False
                    end_time = current_time - (silence_counter * window_size) + padding
                    
                    duration = end_time - start_time
                    if duration >= min_duration:
                        segments.append((start_time, end_time))
            else:
                silence_counter = 0

    if is_active:
        segments.append((start_time, max_time))
        
    # 4. Extract Clips
    extracted_clips = []
    for (t_start, t_end) in segments:
        mask = (full_times >= t_start) & (full_times <= t_end)
        seg_times = full_times[mask]
        seg_units = full_units[mask]
        
        # Shift time to 0
        seg_times = seg_times - t_start 
        
        # Length Normalization (Truncating)
        if target_length is not None:
            valid_mask = seg_times < target_length
            truncated = np.sum(~valid_mask) > 0
            seg_times = seg_times[valid_mask]
            seg_units = seg_units[valid_mask]
        else:
            truncated = False

        extracted_clips.append({
            'times': seg_times,
            'units': seg_units,
            'original_start': t_start,
            'detected_duration': t_end - t_start,
            'is_truncated': truncated
        })
        
    return extracted_clips

def sparse_data_generator_from_dict(X, batch_size, nb_steps, nb_units, max_time, device, shuffle=True):
    """Generates sparse tensor input from spike dictionary."""
    if not isinstance(X, dict):
        raise ValueError(f"Expected X to be a dictionary, got {type(X)}")
    
    firing_times = np.asarray(X['times'], dtype=np.float32)
    units_fired = np.asarray(X['units'], dtype=np.int64)
    
    if batch_size != 1:
        batch_size = 1
    
    # Double check truncation (in case VAD didn't handle it or logic changed)
    valid_mask = firing_times < max_time
    firing_times = firing_times[valid_mask]
    units_fired = units_fired[valid_mask]
    
    time_bins = np.linspace(0, max_time, num=nb_steps)
    times = np.digitize(firing_times, time_bins)
    times = np.clip(times, 0, nb_steps - 1) # Safe clamp
    
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

    # Note: Explicit size is important here to ensure fixed tensor shape
    X_batch = torch.sparse_coo_tensor(i, v, torch.Size([batch_size, nb_steps, nb_units]), device=device)

    return X_batch.to(device=device)

# --- 3. SNN Execution ---

def run_snn(inputs, w1, w2, v1, batch_size, nb_hidden, nb_outputs, nb_steps, alpha, beta, device, dtype):
    """Run SNN forward pass."""
    syn = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
    mem = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)

    spk_rec = []

    # Hidden layer
    out = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
    h1_from_input = torch.einsum("abc,cd->abd", (inputs, w1))
    
    for t in range(nb_steps):
        h1 = h1_from_input[:, t] + torch.einsum("ab,bc->ac", (out, v1))
        mthr = mem - 1.0
        out = spike_fn(mthr)
        rst = out.detach()

        new_syn = alpha * syn + h1
        new_mem = (beta * mem + syn) * (1.0 - rst)
        spk_rec.append(out)
        mem = new_mem
        syn = new_syn

    spk_rec = torch.stack(spk_rec, dim=1)

    # Readout layer
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
    return out_rec, [None, spk_rec]

def predict(checkpoint, x_data, device=None):
    """Make prediction on a single sample."""
    w1, w2, v1 = checkpoint['w1'], checkpoint['w2'], checkpoint['v1']
    if device is None: device = w1.device
    
    # Hyperparameters
    nb_inputs = checkpoint.get('nb_inputs', 700)
    nb_hidden = checkpoint.get('nb_hidden', 200)
    nb_outputs = checkpoint.get('nb_outputs', 10)
    nb_steps = checkpoint.get('nb_steps', 100)
    max_time = checkpoint.get('max_time', 1.4)
    
    # Physics parameters
    time_step = checkpoint.get('time_step', 1e-3)
    tau_mem = checkpoint.get('tau_mem', 10e-3)
    tau_syn = checkpoint.get('tau_syn', 5e-3)
    alpha = checkpoint.get('alpha', float(np.exp(-time_step/tau_syn)))
    beta = checkpoint.get('beta', float(np.exp(-time_step/tau_mem)))
    
    try:
        # Generate sparse tensor (handles padding/truncation)
        x_data_tensor = sparse_data_generator_from_dict(x_data, 1, nb_steps, nb_inputs, max_time, device)
        x_data_dense = x_data_tensor.to_dense()
        
        output, _ = run_snn(
            x_data_dense, w1, w2, v1, 
            1, nb_hidden, nb_outputs, nb_steps, 
            alpha, beta, device, torch.float
        )
        
        m, _ = torch.max(output, 1)  # max over time
        _, am = torch.max(m, 1)      # argmax over units
        return am.detach().cpu().numpy()[0]
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def load_spike_file(filepath):
    """Load .npz file."""
    my_data = np.load(filepath, allow_pickle=True)
    if 'arr_0' in my_data:
        data_array = my_data['arr_0']
        data_times, data_units = data_array[0], data_array[1]
    elif 'times' in my_data and 'units' in my_data:
        data_times, data_units = my_data['times'], my_data['units']
    else:
        raise ValueError("Unknown data format")
        
    return {'times': np.asarray(data_times, dtype=np.float32), 
            'units': np.asarray(data_units, dtype=np.int64)}

# --- 4. Main Execution ---

def main():
    # ---------------------------------------------------------
    # CONFIGURATION
    # ---------------------------------------------------------
    model_path = 'model/best_model_dropout_epoch220_testacc0.852.pth'
    target_file = 'data/longSpikes/long0to9.npz' # 
    vad_target_length = 1.4  # VAD will normalize clips to this length
    # ---------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Model
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    checkpoint = load_model(model_path, device=device)
    print("Model loaded successfully.")
    print("-" * 50)

    # 2. Load Long Audio File
    if not os.path.exists(target_file):
        print(f"Error: Target file not found: {target_file}")
        return
    
    print(f"Loading file: {target_file}")
    try:
        long_audio_data = load_spike_file(target_file)
        print(f"Total Spikes: {len(long_audio_data['times'])}")
        print(f"Duration: {long_audio_data['times'].max():.2f}s")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # 3. Phase 2: VAD Segmentation
    print("\n[Phase 2] Running VAD Segmentation...")
    clips = snn_vad_segmentation(
        long_audio_data['times'], 
        long_audio_data['units'],
        target_length=vad_target_length, # Pass 1.4s here
        threshold_start=100, 
        threshold_end=50
    )
    
    print(f"--> Found {len(clips)} potential digit segments.")
    print("-" * 50)

    # 4. Phase 1: Inference on Segments
    print("\n[Phase 1] Running Inference on Segments...")
    
    final_sequence = []
    
    for i, clip in enumerate(clips, 1):
        print(f"\nProcessing Segment {i}:")
        print(f"  - Original Start: {clip['original_start']:.2f}s")
        print(f"  - Duration: {clip['detected_duration']:.2f}s")
        print(f"  - Spikes: {len(clip['times'])}")
        
        # Predict
        pred_digit = predict(checkpoint, clip, device=device)
        
        if pred_digit is not None:
            print(f"  => Prediction: {pred_digit}")
            final_sequence.append(int(pred_digit))
        else:
            print("  => Prediction Failed")

    print("\n" + "=" * 50)
    print("FINAL RESULT")
    print("=" * 50)
    print(f"Input File: {os.path.basename(target_file)}")
    print(f"Digit Sequence: {final_sequence}")
    print(f"Count: {len(final_sequence)}")
    print("=" * 50)

if __name__ == "__main__":
    main()