import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils import data

# Training configuration
nb_epochs = 600

# Network structure
nb_inputs  = 700
nb_hidden  = 200
nb_outputs = 10

# Time parameters
time_step = 1e-3
nb_steps = 100
max_time = 1.4

batch_size = 256

dtype = torch.float

# Dropout rates
dropout_rate_w1 = 0.1  # Input to hidden dropout
dropout_rate_v1 = 0.1  # Recurrent dropout
dropout_rate_w2 = 0.1  # Hidden to output dropout

# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")     
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Digit names for loading files
digit_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

def load_npz_dataset(data_folder, max_time=1.4, nb_steps=100, nb_units=700):
    """
    Load all npz files from a folder and convert to sparse spike format.
    
    Args:
        data_folder: Path to folder containing npz files
        max_time: Maximum time duration (with zero padding)
        nb_steps: Number of time bins
        nb_units: Number of neuron units
    
    Returns:
        spike_times_list: List of spike times for each sample
        spike_units_list: List of spike units for each sample
        labels_list: List of labels for each sample
    """
    spike_times_list = []
    spike_units_list = []
    labels_list = []
    
    # Get all npz files in the folder
    npz_files = glob.glob(os.path.join(data_folder, '*.npz'))
    
    print(f"Found {len(npz_files)} npz files in {data_folder}")
    
    for npz_file in sorted(npz_files):
        filename = os.path.basename(npz_file)
        
        # Extract label from filename pattern: digit-X
        # Example: noisy_lang-english_speaker-00_trial-47_digit-0.npz
        label = None
        if 'digit-' in filename:
            try:
                # Find the digit after 'digit-'
                digit_start = filename.index('digit-') + 6
                # Extract the number (could be single or double digit)
                digit_str = ''
                for char in filename[digit_start:]:
                    if char.isdigit():
                        digit_str += char
                    else:
                        break
                label = int(digit_str)
                
                # Validate label is in range 0-9
                if label < 0 or label > 9:
                    print(f"Warning: Label {label} out of range for {filename}, skipping")
                    continue
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse digit from {filename}, skipping")
                continue
        
        if label is None:
            print(f"Warning: Could not determine label for {filename}, skipping")
            continue
        
        try:
            # Load data
            data = np.load(npz_file, allow_pickle=True)
            
            if 'arr_0' in data:
                time_data = data['arr_0'][0]
                unit_data = data['arr_0'][1]
            else:
                print(f"Warning: {filename} - arr_0 not found, skipping")
                continue
            
            # Filter out spikes beyond max_time
            valid_mask = time_data < max_time
            time_data = time_data[valid_mask]
            unit_data = unit_data[valid_mask]
            
            # Ensure unit_data is within valid range
            unit_data = np.clip(unit_data.astype(int), 0, nb_units - 1)
            
            # Store the spike data
            spike_times_list.append(time_data)
            spike_units_list.append(unit_data)
            labels_list.append(label)
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    print(f"Successfully loaded {len(labels_list)} samples")
    print(f"Label distribution: {np.bincount(labels_list)}")
    
    return spike_times_list, spike_units_list, np.array(labels_list)


def sparse_data_generator_from_npz(spike_times_list, spike_units_list, labels, 
                                   batch_size, nb_steps, nb_units, max_time, shuffle=True):
    """
    Generate batches of sparse spike tensors from npz data.
    
    Args:
        spike_times_list: List of spike times for each sample
        spike_units_list: List of spike units for each sample
        labels: Array of labels
        batch_size: Batch size
        nb_steps: Number of time steps
        nb_units: Number of input units
        max_time: Maximum time duration
        shuffle: Whether to shuffle the data
    """
    
    number_of_batches = len(labels) // batch_size
    sample_index = np.arange(len(labels))
    
    # Compute discrete time bins
    time_bins = np.linspace(0, max_time, num=nb_steps)
    
    if shuffle:
        np.random.shuffle(sample_index)
    
    counter = 0
    while counter < number_of_batches:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        
        coo = [[], [], []]  # [batch_idx, time_idx, unit_idx]
        
        for bc, idx in enumerate(batch_index):
            # Get spike times and units for this sample
            times = spike_times_list[idx]
            units = spike_units_list[idx]
            
            # Digitize times into bins
            time_indices = np.digitize(times, time_bins)
            time_indices = np.clip(time_indices, 0, nb_steps - 1)
            
            # Create batch indices
            batch = [bc] * len(times)
            
            coo[0].extend(batch)
            coo[1].extend(time_indices)
            coo[2].extend(units)
        
        # Create sparse tensor
        i = torch.LongTensor(coo).to(device)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)
        
        X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size, nb_steps, nb_units])).to(device)
        y_batch = torch.tensor(labels[batch_index], device=device)
        
        yield X_batch.to(device=device), y_batch.to(device=device)
        
        counter += 1


# Initialize network parameters
tau_mem = 10e-3
tau_syn = 5e-3

alpha = float(np.exp(-time_step/tau_syn))
beta = float(np.exp(-time_step/tau_mem))

weight_scale = 0.2

w1 = torch.empty((nb_inputs, nb_hidden), device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(w1, mean=0.0, std=weight_scale/np.sqrt(nb_inputs))

w2 = torch.empty((nb_hidden, nb_outputs), device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(w2, mean=0.0, std=weight_scale/np.sqrt(nb_hidden))

v1 = torch.empty((nb_hidden, nb_hidden), device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(v1, mean=0.0, std=weight_scale/np.sqrt(nb_hidden))

print("Network initialization done")


def plot_voltage_traces(mem, spk=None, dim=(3,5), spike_height=5):
    gs = GridSpec(*dim)
    if spk is not None:
        dat = 1.0*mem
        dat[spk>0.0] = spike_height
        dat = dat.detach().cpu().numpy()
    else:
        dat = mem.detach().cpu().numpy()
    for i in range(np.prod(dim)):
        if i==0: a0=ax=plt.subplot(gs[i])
        else: ax=plt.subplot(gs[i], sharey=a0)
        ax.plot(dat[i])
        ax.axis("off")


class SurrGradSpike(torch.autograd.Function):
    """
    Spiking nonlinearity with surrogate gradient.
    """
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
        grad = grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0)**2
        return grad

spike_fn = SurrGradSpike.apply


def apply_weight_dropout(weight, dropout_rate, training=True):
    """Apply inverted dropout to weight matrix."""
    if not training or dropout_rate == 0.0:
        return weight
    
    keep_prob = 1.0 - dropout_rate
    mask = (torch.rand_like(weight) < keep_prob).float()
    dropped_weight = weight * mask / keep_prob
    
    return dropped_weight


def run_snn(inputs, training=True):
    """Run the spiking neural network with optional dropout."""
    
    # Apply dropout to weight matrices
    w1_dropped = apply_weight_dropout(w1, dropout_rate_w1, training)
    v1_dropped = apply_weight_dropout(v1, dropout_rate_v1, training)
    w2_dropped = apply_weight_dropout(w2, dropout_rate_w2, training)
    
    syn = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
    mem = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)

    mem_rec = []
    spk_rec = []

    # Compute hidden layer activity
    out = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
    h1_from_input = torch.einsum("abc,cd->abd", (inputs, w1_dropped))
    
    for t in range(nb_steps):
        h1 = h1_from_input[:, t] + torch.einsum("ab,bc->ac", (out, v1_dropped))
        mthr = mem - 1.0
        out = spike_fn(mthr)
        rst = out.detach()

        new_syn = alpha*syn + h1
        new_mem = (beta*mem + syn)*(1.0 - rst)

        mem_rec.append(mem)
        spk_rec.append(out)
        
        mem = new_mem
        syn = new_syn

    mem_rec = torch.stack(mem_rec, dim=1)
    spk_rec = torch.stack(spk_rec, dim=1)

    # Readout layer
    h2 = torch.einsum("abc,cd->abd", (spk_rec, w2_dropped))
    flt = torch.zeros((batch_size, nb_outputs), device=device, dtype=dtype)
    out = torch.zeros((batch_size, nb_outputs), device=device, dtype=dtype)
    out_rec = [out]
    
    for t in range(nb_steps):
        new_flt = alpha*flt + h2[:, t]
        new_out = beta*out + flt

        flt = new_flt
        out = new_out

        out_rec.append(out)

    out_rec = torch.stack(out_rec, dim=1)
    other_recs = [mem_rec, spk_rec]
    return out_rec, other_recs


def compute_classification_accuracy(spike_times_list, spike_units_list, labels):
    """Computes classification accuracy on supplied data in batches."""
    accs = []
    for x_local, y_local in sparse_data_generator_from_npz(
        spike_times_list, spike_units_list, labels, 
        batch_size, nb_steps, nb_inputs, max_time, shuffle=False):
        
        output, _ = run_snn(x_local.to_dense(), training=False)
        m, _ = torch.max(output, 1)  # max over time
        _, am = torch.max(m, 1)      # argmax over output units
        tmp = np.mean((y_local == am).detach().cpu().numpy())
        accs.append(tmp)
    return np.mean(accs)


def save_model(filepath, w1, w2, v1, loss_hist=None, **kwargs):
    """Save the trained model weights and configuration."""
    checkpoint = {
        'w1': w1.cpu().detach().clone(),
        'w2': w2.cpu().detach().clone(),
        'v1': v1.cpu().detach().clone(),
        'loss_hist': loss_hist if loss_hist is not None else [],
    }
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")


def train(train_spike_times, train_spike_units, train_labels,
          test_spike_times, test_spike_units, test_labels,
          lr=2e-4, nb_epochs=10, eval_interval=10):
    
    params = [w1, w2, v1]
    optimizer = torch.optim.Adamax(params, lr=lr, betas=(0.9, 0.999))

    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()
    
    loss_hist = []
    train_acc_hist = []
    test_acc_hist = []
    lr_hist = []
    
    best_test_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    patience_limit = 50
    
    os.makedirs('model', exist_ok=True)
    
    current_lr = lr
    
    for e in range(nb_epochs):
        # Dynamic learning rate scheduling
        if e == 150:
            current_lr = lr / 2
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            print(f"\n>>> Learning rate reduced to {current_lr:.2e} at epoch {e+1}")
        elif e == 300:
            current_lr = lr / 4
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            print(f"\n>>> Learning rate reduced to {current_lr:.2e} at epoch {e+1}")
        
        lr_hist.append(current_lr)
        
        local_loss = []
        for x_local, y_local in sparse_data_generator_from_npz(
            train_spike_times, train_spike_units, train_labels,
            batch_size, nb_steps, nb_inputs, max_time):
            
            output, recs = run_snn(x_local.to_dense(), training=True)
            _, spks = recs
            m, _ = torch.max(output, 1)
            log_p_y = log_softmax_fn(m)
            
            # Regularizer loss
            reg_loss = 2e-6*torch.sum(spks)
            reg_loss += 2e-6*torch.mean(torch.sum(torch.sum(spks, dim=0), dim=0)**2)
            
            loss_val = loss_fn(log_p_y, y_local) + reg_loss

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            local_loss.append(loss_val.item())
        
        mean_loss = np.mean(local_loss)
        loss_hist.append(mean_loss)
        
        # Evaluate every eval_interval epochs
        if (e + 1) % eval_interval == 0 or e == nb_epochs - 1:
            train_acc = compute_classification_accuracy(train_spike_times, train_spike_units, train_labels)
            test_acc = compute_classification_accuracy(test_spike_times, test_spike_units, test_labels)
            train_acc_hist.append((e+1, train_acc))
            test_acc_hist.append((e+1, test_acc))
            
            print(f"Epoch {e+1}: loss={mean_loss:.5f}, train_acc={train_acc:.3f}, test_acc={test_acc:.3f}, lr={current_lr:.2e}")
            
            # Save best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = e + 1
                patience_counter = 0
                
                best_model_path = f'model/best_model_npz_epoch{e+1}_testacc{test_acc:.3f}.pth'
                save_model(
                    best_model_path,
                    w1, w2, v1,
                    loss_hist=loss_hist,
                    train_acc_hist=train_acc_hist,
                    test_acc_hist=test_acc_hist,
                    epoch=e+1,
                    nb_inputs=nb_inputs,
                    nb_hidden=nb_hidden,
                    nb_outputs=nb_outputs,
                    dropout_rate_w1=dropout_rate_w1,
                    dropout_rate_v1=dropout_rate_v1,
                    dropout_rate_w2=dropout_rate_w2,
                    tau_mem=tau_mem,
                    tau_syn=tau_syn,
                    time_step=time_step,
                    nb_steps=nb_steps,
                    max_time=max_time,
                    batch_size=batch_size,
                    weight_scale=weight_scale,
                    alpha=alpha,
                    beta=beta,
                    learning_rate=current_lr
                )
                print(f"*** New best model saved! Test accuracy: {test_acc:.3f} ***")
            else:
                patience_counter += eval_interval
                
            # Early stopping check
            if patience_counter >= patience_limit:
                print(f"\n>>> Early stopping triggered at epoch {e+1}")
                print(f">>> No improvement for {patience_limit} epochs")
                break
        else:
            print(f"Epoch {e+1}: loss={mean_loss:.5f}, lr={current_lr:.2e}")
    
    print(f"\n=== Training Complete ===")
    print(f"Best test accuracy: {best_test_acc:.3f} at epoch {best_epoch}")
    print(f"Final learning rate: {current_lr:.2e}")
    
    return loss_hist, train_acc_hist, test_acc_hist


# ========== MAIN EXECUTION ==========

# Set your data folders here
TRAIN_FOLDER = 'data/noisy_spikes_v4_training'  # Replace with your training data folder path
TEST_FOLDER = 'data/noisy_spikes_v4_testing'    # Replace with your test data folder path

print("=" * 70)
print("Loading Training Data...")
print("=" * 70)
train_spike_times, train_spike_units, train_labels = load_npz_dataset(
    TRAIN_FOLDER, max_time=max_time, nb_steps=nb_steps, nb_units=nb_inputs
)

print("\n" + "=" * 70)
print("Loading Test Data...")
print("=" * 70)
test_spike_times, test_spike_units, test_labels = load_npz_dataset(
    TEST_FOLDER, max_time=max_time, nb_steps=nb_steps, nb_units=nb_inputs
)

# Print configuration
print("\n" + "=" * 70)
print("Training Configuration:")
print(f"  Epochs:                {nb_epochs}")
print(f"  Hidden units:          {nb_hidden}")
print(f"  Batch size:            {batch_size}")
print(f"  Learning rate:         2e-4")
print(f"  Training samples:      {len(train_labels)}")
print(f"  Test samples:          {len(test_labels)}")
print("-" * 70)
print("Dropout Configuration:")
print(f"  W1 (input-hidden):     {dropout_rate_w1:.1%}")
print(f"  V1 (hidden-hidden):    {dropout_rate_v1:.1%} (recurrent)")
print(f"  W2 (hidden-output):    {dropout_rate_w2:.1%}")
print("=" * 70)

# Train the model
loss_hist, train_acc_hist, test_acc_hist = train(
    train_spike_times, train_spike_units, train_labels,
    test_spike_times, test_spike_units, test_labels,
    lr=2e-4,
    nb_epochs=nb_epochs,
    eval_interval=10
)

# Print accuracy history
print("\n=== Accuracy History ===")
for epoch, train_acc in train_acc_hist:
    test_acc = [acc for e, acc in test_acc_hist if e == epoch][0]
    print(f"Epoch {epoch}: Train Acc = {train_acc:.3f}, Test Acc = {test_acc:.3f}")

# Generate visualization with test data
print("\nGenerating visualizations...")
test_batch_gen = sparse_data_generator_from_npz(
    test_spike_times, test_spike_units, test_labels,
    batch_size, nb_steps, nb_inputs, max_time, shuffle=False
)
x_batch, y_batch = next(test_batch_gen)

output, other_recordings = run_snn(x_batch.to_dense(), training=False)
mem_rec, spk_rec = other_recordings

# Save voltage traces
fig = plt.figure(dpi=100)
plot_voltage_traces(output)
plt.savefig('voltage_traces_npz.png')
plt.close()
print("Saved: voltage_traces_npz.png")

# Save spike activity
nb_plt = 4
gs = GridSpec(1, nb_plt)
fig = plt.figure(figsize=(7, 3), dpi=150)
for i in range(nb_plt):
    plt.subplot(gs[i])
    plt.imshow(spk_rec[i].detach().cpu().numpy().T, cmap=plt.cm.gray_r, origin="lower")
    if i == 0:
        plt.xlabel("Time")
        plt.ylabel("Units")
    sns.despine()

plt.savefig('spike_activity_npz.png')
plt.close()
print("Saved: spike_activity_npz.png")

print("\nTraining complete! Best model and visualizations saved.")