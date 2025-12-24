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
from utils import get_shd_dataset


class SurrGradSpike(torch.autograd.Function):
    """
    Spiking nonlinearity which also implements the surrogate gradient.
    By subclassing torch.autograd.Function, we can use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid as done in Zenke & Ganguli (2018).
    """
    
    scale = 100.0  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which 
        we need to later backpropagate our error signals.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the 
        surrogate gradient of the loss with respect to the input. 
        Here we use the normalized negative part of a fast sigmoid 
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
        return grad


# Spike function
spike_fn = SurrGradSpike.apply


def load_model(filepath, device=None):
    """
    Load a saved model checkpoint.
    
    Args:
        filepath: Path to the saved model file
        device: Device to load the model on (default: uses current device)
    
    Returns:
        Dictionary containing weights and saved hyperparameters
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Clear any previous CUDA errors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Check for and clear any pending CUDA errors
        try:
            torch.cuda.synchronize()
        except RuntimeError as e:
            print(f"Warning: Clearing CUDA error: {e}")
            torch.cuda.empty_cache()
    
    # Load checkpoint on CPU first to avoid CUDA errors during loading
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    
    # Load weights and move to device with error handling
    try:
        if device.type == 'cuda':
            # For CUDA, move tensors one at a time and check for errors
            torch.cuda.synchronize()  # Ensure no pending operations
            w1 = checkpoint['w1'].to(device).requires_grad_(True)
            torch.cuda.synchronize()  # Check for errors after each move
            w2 = checkpoint['w2'].to(device).requires_grad_(True)
            torch.cuda.synchronize()
            v1 = checkpoint['v1'].to(device).requires_grad_(True)
            torch.cuda.synchronize()
        else:
            # For CPU, no need for synchronization
            w1 = checkpoint['w1'].to(device).requires_grad_(True)
            w2 = checkpoint['w2'].to(device).requires_grad_(True)
            v1 = checkpoint['v1'].to(device).requires_grad_(True)
    except RuntimeError as e:
        print(f"Error moving tensors to {device}: {e}")
        print("Attempting to load on CPU instead...")
        device = torch.device('cpu')
        w1 = checkpoint['w1'].to(device).requires_grad_(True)
        w2 = checkpoint['w2'].to(device).requires_grad_(True)
        v1 = checkpoint['v1'].to(device).requires_grad_(True)
    
    print(f"Model loaded from {filepath}")
    print(f"Loaded weights: w1 {w1.shape}, w2 {w2.shape}, v1 {v1.shape}")
    print(f"Device: {device}")
    
    # Return weights and all saved hyperparameters
    result = {
        'w1': w1,
        'w2': w2,
        'v1': v1,
        'loss_hist': checkpoint.get('loss_hist', [])
    }
    # Add all other saved parameters
    for key, value in checkpoint.items():
        if key not in ['w1', 'w2', 'v1', 'loss_hist']:
            result[key] = value
    
    return result


def sparse_data_generator_from_dict(X, batch_size, nb_steps, nb_units, max_time, device, shuffle=True):
    """
    This generator takes a spike dataset and generates spiking network input as sparse tensors. 

    Args:
        X: The data dictionary with 'times' and 'units' keys (single sample)
        batch_size: Batch size (should be 1 for single sample inference)
        nb_steps: Number of time steps
        nb_units: Number of input units
        max_time: Maximum time value
        device: Device to place tensors on
        shuffle: Whether to shuffle (not used in current implementation)
    
    Returns:
        Sparse tensor of shape (batch_size, nb_steps, nb_units)
    """
    # Validate input data format
    if not isinstance(X, dict):
        raise ValueError(f"Expected X to be a dictionary, got {type(X)}")
    if 'times' not in X or 'units' not in X:
        raise ValueError("X must contain 'times' and 'units' keys")
    
    # compute discrete firing times
    firing_times = np.asarray(X['times'], dtype=np.float32)
    units_fired = np.asarray(X['units'], dtype=np.int64)
    
    # Validate data lengths match
    if len(firing_times) != len(units_fired):
        raise ValueError(f"Length mismatch: times has {len(firing_times)} elements, units has {len(units_fired)}")
    
    # Validate batch_size for single sample
    if batch_size != 1:
        print(f"Warning: batch_size={batch_size} but processing single sample. Setting batch_size=1")
        batch_size = 1
    
    
    # over max_time Spike (Truncating)
    valid_mask = firing_times < max_time
    firing_times = firing_times[valid_mask]
    units_fired = units_fired[valid_mask]  # filter unit
    
    time_bins = np.linspace(0, max_time, num=nb_steps)
    
    # Digitize times and clamp to valid range [0, nb_steps-1]
    times = np.digitize(firing_times, time_bins)
    # np.digitize returns indices in [0, nb_steps], so clamp to [0, nb_steps-1]
    #times = np.clip(times, 0, nb_steps - 1)
    
    # Validate and clamp units to valid range [0, nb_units-1]
    if np.any(units_fired < 0) or np.any(units_fired >= nb_units):
        invalid_count = np.sum((units_fired < 0) | (units_fired >= nb_units))
        print(f"Warning: {invalid_count} unit indices out of bounds [0, {nb_units-1}]. Clamping to valid range.")
        units_fired = np.clip(units_fired, 0, nb_units - 1)
    
    units = units_fired
    batch = [0 for _ in range(len(times))]
    
    coo = [ [] for i in range(3) ]
    coo[0].extend(batch)
    coo[1].extend(times)
    coo[2].extend(units)

    i = torch.LongTensor(coo).to(device)
    v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)

    # Use the newer sparse_coo_tensor instead of deprecated FloatTensor
    X_batch = torch.sparse_coo_tensor(i, v, torch.Size([batch_size, nb_steps, nb_units]), device=device)

    return X_batch.to(device=device)


def run_snn(inputs, w1, w2, v1, batch_size, nb_hidden, nb_outputs, nb_steps, alpha, beta, device, dtype):
    """
    Run the spiking neural network forward pass.
    
    Args:
        inputs: Input tensor of shape (batch_size, nb_steps, nb_inputs)
        w1: Input-to-hidden weight matrix
        w2: Hidden-to-output weight matrix
        v1: Recurrent weight matrix
        batch_size: Batch size
        nb_hidden: Number of hidden units
        nb_outputs: Number of output units
        nb_steps: Number of time steps
        alpha: Synaptic decay factor
        beta: Membrane decay factor
        device: Device to run on
        dtype: Data type
    
    Returns:
        output: Output tensor
        other_recs: List containing [mem_rec, spk_rec]
    """
    syn = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
    mem = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)

    mem_rec = []
    spk_rec = []

    # Compute hidden layer activity
    out = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
    h1_from_input = torch.einsum("abc,cd->abd", (inputs, w1))
    for t in range(nb_steps):
        h1 = h1_from_input[:, t] + torch.einsum("ab,bc->ac", (out, v1))
        mthr = mem - 1.0
        out = spike_fn(mthr)
        rst = out.detach()  # We do not want to backprop through the reset

        new_syn = alpha * syn + h1
        new_mem = (beta * mem + syn) * (1.0 - rst)

        mem_rec.append(mem)
        spk_rec.append(out)
        
        mem = new_mem
        syn = new_syn

    mem_rec = torch.stack(mem_rec, dim=1)
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
    other_recs = [mem_rec, spk_rec]
    return out_rec, other_recs


def predict(checkpoint, x_data, y_data=None, device=None, batch_size=None):
    """
    Make predictions using the loaded model checkpoint.
    
    Args:
        checkpoint: Dictionary returned from load_model() containing weights and hyperparameters
        x_data: Input data dictionary with 'times' and 'units' keys (single sample)
        y_data: Optional labels (not used in current implementation)
        device: Device to run on (default: uses checkpoint device)
        batch_size: Batch size for inference (default: 1 for single sample)
    
    Returns:
        predictions: Numpy array of predicted class indices
    """
    try:
        w1 = checkpoint['w1']
        w2 = checkpoint['w2']
        v1 = checkpoint['v1']
    except KeyError as e:
        raise ValueError(f"Checkpoint missing required key: {e}")
    
    if device is None:
        device = w1.device
    
    # Get hyperparameters from checkpoint with defaults
    nb_inputs = checkpoint.get('nb_inputs', 700)
    nb_hidden = checkpoint.get('nb_hidden', 200)
    nb_outputs = checkpoint.get('nb_outputs', 10)
    nb_steps = checkpoint.get('nb_steps', 100)
    max_time = checkpoint.get('max_time', 1.4)
    
    # For single sample inference, use batch_size=1
    # Allow override via parameter, but default to 1 for single sample
    if batch_size is None:
        batch_size = 1  # Single sample inference
    
    # Validate batch_size
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    
    # Compute alpha and beta from tau values
    time_step = checkpoint.get('time_step', 1e-3)
    tau_mem = checkpoint.get('tau_mem', 10e-3)
    tau_syn = checkpoint.get('tau_syn', 5e-3)
    alpha = checkpoint.get('alpha', float(np.exp(-time_step/tau_syn)))
    beta = checkpoint.get('beta', float(np.exp(-time_step/tau_mem)))
    
    dtype = torch.float
    print(f'batch_size: {batch_size}, nb_steps: {nb_steps}, nb_inputs: {nb_inputs}, max_time: {max_time}')

    try:
        x_data = sparse_data_generator_from_dict(x_data, batch_size, nb_steps, nb_inputs, max_time, device)
        print(f'x_data.shape: {x_data.shape}')
        input()
        x_data = x_data.to_dense()
        print(f'x_data.shape: {x_data.shape}')
    except Exception as e:
        raise RuntimeError(f"Error processing input data: {e}") from e
    
    try:
        output, _ = run_snn(
            x_data, w1, w2, v1, 
            batch_size, nb_hidden, nb_outputs, nb_steps, 
            alpha, beta, device, dtype
        )
        m, _ = torch.max(output, 1)  # max over time
        _, am = torch.max(m, 1)       # argmax over output units
        predictions = am.detach().cpu().numpy()
        return predictions
    except RuntimeError as e:
        if 'CUDA' in str(e) or 'cuda' in str(e):
            print(f"CUDA error during inference: {e}")
            print("Attempting to run on CPU...")
            device = torch.device('cpu')
            w1 = w1.to(device)
            w2 = w2.to(device)
            v1 = v1.to(device)
            x_data = x_data.to(device)
            output, _ = run_snn(
                x_data, w1, w2, v1, 
                batch_size, nb_hidden, nb_outputs, nb_steps, 
                alpha, beta, device, dtype
            )
            m, _ = torch.max(output, 1)
            _, am = torch.max(m, 1)
            predictions = am.detach().cpu().numpy()
            return predictions
        else:
            raise


def load_spike_file(filepath):
    """
    Load spike data from .npz file.
    
    Args:
        filepath: Path to .npz file
    
    Returns:
        Dictionary with 'times' and 'units' keys
    """
    my_data = np.load(filepath, allow_pickle=True)
    
    # Handle different possible data structures
    if 'arr_0' in my_data:
        data_array = my_data['arr_0']
        if isinstance(data_array, np.ndarray) and len(data_array) >= 2:
            data_times = data_array[0]
            data_units = data_array[1]
        else:
            raise ValueError(f"Unexpected data structure in 'arr_0': {type(data_array)}")
    elif 'times' in my_data and 'units' in my_data:
        data_times = my_data['times']
        data_units = my_data['units']
    else:
        raise ValueError(f"Unknown data format. Available keys: {list(my_data.keys())}")
    
    # Validate data
    data_times = np.asarray(data_times, dtype=np.float32)
    data_units = np.asarray(data_units, dtype=np.int64)
    
    if len(data_times) == 0 or len(data_units) == 0:
        raise ValueError("Empty data arrays")
    if len(data_times) != len(data_units):
        raise ValueError(f"Length mismatch: times={len(data_times)}, units={len(data_units)}")
    
    return {
        'times': data_times,
        'units': data_units
    }


def main():
    """Main execution function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("=" * 70)
    
    # Load the model
    checkpoint = load_model('model/best_model_epoch130_testacc0.941.pth', device=device)
    print("=" * 70)
    
    # Process all files in myData/spikes/myOneToTen
    spikes_dir = 'data/test_npz'
    
    if not os.path.exists(spikes_dir):
        print(f"Error: Directory '{spikes_dir}' does not exist!")
        return
    
    # # Find all .npz files
    # spike_files = sorted(Path(spikes_dir).glob("*.npz"))
    target_count = 20
    spike_files = []

    # 使用 iterator，不要用 list() 轉成列表，這樣就不會一次載入所有檔案
    iterator = Path(spikes_dir).glob("*.npz")

    for i, file_path in enumerate(iterator):
        if i < target_count:
            # 前 20 個直接放入
            spike_files.append(file_path)
        else:
            # 第 21 個開始，以 20/(i+1) 的機率替換掉原本的某一個
            # 這是數學上證明均勻隨機的
            j = random.randint(0, i)
            if j < target_count:
                spike_files[j] = file_path
    
    if not spike_files:
        print(f"No .npz files found in '{spikes_dir}'")
        return
    
    print(f"\nFound {len(spike_files)} spike file(s) in {spikes_dir}")
    print("=" * 70)
    
    results = []
    success_count = 0
    fail_count = 0
    
    # Process each file
    for i, spike_file in enumerate(spike_files, 1):
        filename = spike_file.name
        print(f"\n[{i}/{len(spike_files)}] Processing: {filename}")
        print("-" * 70)
        
        try:
            # Load spike data
            input_data = load_spike_file(str(spike_file))
            
            print(f"Data loaded: {len(input_data['times'])} spikes")
            print(f"Time range: [{input_data['times'].min():.4f}, {input_data['times'].max():.4f}]")
            print(f"Unit range: [{input_data['units'].min()}, {input_data['units'].max()}]")
            
            # Make prediction
            prediction = predict(checkpoint, input_data, device=device, batch_size=1)
            predicted_class = int(prediction[0])
            
            results.append({
                'filename': filename,
                'prediction': predicted_class,
                'spikes': len(input_data['times']),
                'status': 'success'
            })
            
            print(f"✓ Prediction: Class {predicted_class}")
            success_count += 1
            
        except FileNotFoundError as e:
            print(f"✗ Error: {e}")
            results.append({
                'filename': filename,
                'prediction': None,
                'status': 'file_not_found',
                'error': str(e)
            })
            fail_count += 1
        except ValueError as e:
            print(f"✗ Error: Invalid data format - {e}")
            results.append({
                'filename': filename,
                'prediction': None,
                'status': 'data_error',
                'error': str(e)
            })
            fail_count += 1
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'filename': filename,
                'prediction': None,
                'status': 'error',
                'error': str(e)
            })
            fail_count += 1
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files processed: {len(spike_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print()
    print("Results:")
    print("-" * 70)
    print(f"{'Filename':<25} {'Prediction':<12} {'Spikes':<10} {'Status'}")
    print("-" * 70)
    
    for result in results:
        if result['status'] == 'success':
            print(f"{result['filename']:<25} {result['prediction']:<12} {result['spikes']:<10} ✓")
        else:
            status = result.get('status', 'error')
            error = result.get('error', 'Unknown error')[:30]
            print(f"{result['filename']:<25} {'N/A':<12} {'N/A':<10} ✗ ({status})")
    
    print("=" * 70)


if __name__ == "__main__":
    main()

