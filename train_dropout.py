import os
import h5py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import torch
import torch.nn as nn
import torchvision
from torch.utils import data

#epoch
nb_epochs = 600

# The coarse network structure and the time steps are dicated by the SHD dataset. 
nb_inputs  = 700
nb_hidden  = 200
nb_outputs = 10

time_step = 1e-3
nb_steps = 100
max_time = 1.2

batch_size = 256

dtype = torch.float

# Dropout rates
dropout_rate_w1 = 0.1  # Input to hidden dropout
dropout_rate_v1 = 0.1  # Recurrent dropout (typically higher)
dropout_rate_w2 = 0.1  # Hidden to output dropout

# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")     
else:
    device = torch.device("cpu")
    
    
# Here we load the Dataset
train_file = h5py.File("data/hdspikes/shd_train.h5", 'r')
test_file = h5py.File("data/hdspikes/shd_test.h5", 'r')

x_train_full = train_file['spikes']
y_train_full = train_file['labels']
x_test_full = test_file['spikes']
y_test_full = test_file['labels']

# 
def filter_labels_0_9(x_data, y_data):
    y_array = np.array(y_data)
    valid_indices = np.where(y_array < 10)[0]
    print(f"Total samples: {len(y_array)}, Valid samples (0-9): {len(valid_indices)}")
    return valid_indices

train_indices = filter_labels_0_9(x_train_full, y_train_full)
test_indices = filter_labels_0_9(x_test_full, y_test_full)

# 
y_train = np.array(y_train_full)[train_indices]
y_test = np.array(y_test_full)[test_indices]

print(f"Training samples with labels 0-9: {len(train_indices)}")
print(f"Test samples with labels 0-9: {len(test_indices)}")

def sparse_data_generator_from_hdf5_spikes(X, y, valid_indices, batch_size, nb_steps, nb_units, max_time, shuffle=True):
    """ This generator takes a spike dataset and generates spiking network input as sparse tensors. 
    
    Args:
        X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
        y: The labels (already filtered)
        valid_indices: Indices of samples with labels 0-9
    """

    labels_ = np.array(y, dtype=int)
    number_of_batches = len(labels_)//batch_size
    sample_index = np.arange(len(labels_))

    # compute discrete firing times
    firing_times = X['times']
    units_fired = X['units']
    
    time_bins = np.linspace(0, max_time, num=nb_steps)

    if shuffle:
        np.random.shuffle(sample_index)

    total_batch_count = 0
    counter = 0
    while counter<number_of_batches:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]

        coo = [ [] for i in range(3) ]
        for bc, idx in enumerate(batch_index):
            # 
            original_idx = valid_indices[idx]
            times = np.digitize(firing_times[original_idx], time_bins)
            units = units_fired[original_idx]
            batch = [bc for _ in range(len(times))]
            
            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(units)

        i = torch.LongTensor(coo).to(device)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)
    
        X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size,nb_steps,nb_units])).to(device)
        y_batch = torch.tensor(labels_[batch_index], device=device)

        yield X_batch.to(device=device), y_batch.to(device=device)

        counter += 1

tau_mem = 10e-3
tau_syn = 5e-3

alpha   = float(np.exp(-time_step/tau_syn))
beta    = float(np.exp(-time_step/tau_mem))

weight_scale = 0.2

w1 = torch.empty((nb_inputs, nb_hidden),  device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(w1, mean=0.0, std=weight_scale/np.sqrt(nb_inputs))

w2 = torch.empty((nb_hidden, nb_outputs), device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(w2, mean=0.0, std=weight_scale/np.sqrt(nb_hidden))

v1 = torch.empty((nb_hidden, nb_hidden), device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(v1, mean=0.0, std=weight_scale/np.sqrt(nb_hidden))

print("init done")

def plot_voltage_traces(mem, spk=None, dim=(3,5), spike_height=5):
    gs=GridSpec(*dim)
    if spk is not None:
        dat = 1.0*mem
        dat[spk>0.0] = spike_height
        dat = dat.detach().cpu().numpy()
    else:
        dat = mem.detach().cpu().numpy()
    for i in range(np.prod(dim)):
        if i==0: a0=ax=plt.subplot(gs[i])
        else: ax=plt.subplot(gs[i],sharey=a0)
        ax.plot(dat[i])
        ax.axis("off")

from IPython.display import clear_output

def live_plot(loss):
    if len(loss) == 1:
        return
    clear_output(wait=True)
    plt.figure(figsize=(3,2), dpi=150)
    ax = plt.gca()
    ax.plot(range(1, len(loss) + 1), loss)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.xaxis.get_major_locator().set_params(integer=True)
    sns.despine()
    plt.show()
    plt.close()

#............train...............#

class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements 
    the surrogate gradient. By subclassing torch.autograd.Function, 
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid 
    as this was done in Zenke & Ganguli (2018).
    """
    
    scale = 100.0 # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which 
        we need to later backpropagate our error signals. To achieve this we use the 
        ctx.save_for_backward method.
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
    
# here we overwrite our naive spike function by the "SurrGradSpike" nonlinearity which implements a surrogate gradient
spike_fn  = SurrGradSpike.apply


def apply_weight_dropout(weight, dropout_rate, training=True):
    """
    Apply inverted dropout to weight matrix.
    
    Args:
        weight: Weight tensor to apply dropout to
        dropout_rate: Probability of dropping a connection (0 to 1)
        training: If True, apply dropout; if False, return original weight
    
    Returns:
        Dropped weight matrix (scaled during training)
    """
    if not training or dropout_rate == 0.0:
        return weight
    
    # Generate dropout mask: 1 = keep, 0 = drop
    keep_prob = 1.0 - dropout_rate
    mask = (torch.rand_like(weight) < keep_prob).float()
    
    # Inverted dropout: scale during training so no scaling needed at test time
    dropped_weight = weight * mask / keep_prob
    
    return dropped_weight


def run_snn(inputs, training=True):
    """
    Run the spiking neural network with optional dropout.
    
    Args:
        inputs: Input spike tensor
        training: If True, apply dropout; if False, no dropout
    
    Returns:
        output recordings and other recordings (mem, spk)
    """
    
    # Apply dropout to weight matrices (once per batch)
    # Each sample in the batch gets the same dropout mask (temporal consistency)
    w1_dropped = apply_weight_dropout(w1, dropout_rate_w1, training)
    v1_dropped = apply_weight_dropout(v1, dropout_rate_v1, training)
    w2_dropped = apply_weight_dropout(w2, dropout_rate_w2, training)
    
    syn = torch.zeros((batch_size,nb_hidden), device=device, dtype=dtype)
    mem = torch.zeros((batch_size,nb_hidden), device=device, dtype=dtype)

    mem_rec = []
    spk_rec = []

    # Compute hidden layer activity with dropout on w1
    out = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
    h1_from_input = torch.einsum("abc,cd->abd", (inputs, w1_dropped))
    
    for t in range(nb_steps):
        # Use dropout on both feedforward (w1) and recurrent (v1) connections
        h1 = h1_from_input[:,t] + torch.einsum("ab,bc->ac", (out, v1_dropped))
        mthr = mem-1.0
        out = spike_fn(mthr)
        rst = out.detach() # We do not want to backprop through the reset

        new_syn = alpha*syn + h1
        new_mem = (beta*mem + syn)*(1.0-rst)

        mem_rec.append(mem)
        spk_rec.append(out)
        
        mem = new_mem
        syn = new_syn

    mem_rec = torch.stack(mem_rec,dim=1)
    spk_rec = torch.stack(spk_rec,dim=1)

    # Readout layer with dropout on w2
    h2 = torch.einsum("abc,cd->abd", (spk_rec, w2_dropped))
    flt = torch.zeros((batch_size,nb_outputs), device=device, dtype=dtype)
    out = torch.zeros((batch_size,nb_outputs), device=device, dtype=dtype)
    out_rec = [out]
    
    for t in range(nb_steps):
        new_flt = alpha*flt + h2[:,t]
        new_out = beta*out + flt

        flt = new_flt
        out = new_out

        out_rec.append(out)

    out_rec = torch.stack(out_rec,dim=1)
    other_recs = [mem_rec, spk_rec]
    return out_rec, other_recs


def compute_classification_accuracy(x_data, y_data, valid_indices):
    """ Computes classification accuracy on supplied data in batches. """
    accs = []
    for x_local, y_local in sparse_data_generator_from_hdf5_spikes(x_data, y_data, valid_indices, batch_size, nb_steps, nb_inputs, max_time, shuffle=False):
        # Evaluation mode: no dropout
        output, _ = run_snn(x_local.to_dense(), training=False)
        m, _ = torch.max(output, 1) # max over time
        _, am = torch.max(m, 1)      # argmax over output units
        tmp = np.mean((y_local==am).detach().cpu().numpy()) # compare to labels
        accs.append(tmp)
    return np.mean(accs)


def save_model(filepath, w1, w2, v1, loss_hist=None, **kwargs):
    """
    Save the trained model weights and configuration.
    
    Args:
        filepath: Path where to save the model (e.g., 'model_checkpoint.pth')
        w1, w2, v1: Model weight tensors
        loss_hist: Training loss history (optional)
        **kwargs: Additional hyperparameters to save (e.g., nb_inputs, nb_hidden, etc.)
    """
    checkpoint = {
        'w1': w1.cpu().detach().clone(),  # Clone to avoid issues
        'w2': w2.cpu().detach().clone(),
        'v1': v1.cpu().detach().clone(),
        'loss_hist': loss_hist if loss_hist is not None else [],
    }
    # Add any additional hyperparameters
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")


def train(x_train, y_train, train_idx, x_test, y_test, test_idx, lr=1e-3, nb_epochs=10, eval_interval=10):
    
    params = [w1,w2,v1]
    optimizer = torch.optim.Adamax(params, lr=lr, betas=(0.9,0.999))

    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()
    
    loss_hist = []
    train_acc_hist = []
    test_acc_hist = []
    lr_hist = []
    
    best_test_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    patience_limit = 50  # Early stopping patience
    
    # Create model directory
    os.makedirs('model', exist_ok=True)
    
    # Initial learning rate
    current_lr = lr
    
    for e in range(nb_epochs):
        # Dynamic learning rate scheduling
        if e == 200:
            current_lr = lr / 2  # Reduce to 1e-4
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            print(f"\n>>> Learning rate reduced to {current_lr:.2e} at epoch {e+1}")
        elif e == 400:
            current_lr = lr / 4  # Reduce to 5e-5
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            print(f"\n>>> Learning rate reduced to {current_lr:.2e} at epoch {e+1}")
        
        lr_hist.append(current_lr)
        
        local_loss = []
        for x_local, y_local in sparse_data_generator_from_hdf5_spikes(x_train, y_train, train_idx, batch_size, nb_steps, nb_inputs, max_time):
            # Training mode: dropout is active
            output, recs = run_snn(x_local.to_dense(), training=True)
            _, spks = recs
            m, _ = torch.max(output, 1)
            log_p_y = log_softmax_fn(m)
            
            # Here we set up our regularizer loss
            reg_loss = 2e-6*torch.sum(spks) # L1 loss on total number of spikes
            reg_loss += 2e-6*torch.mean(torch.sum(torch.sum(spks,dim=0),dim=0)**2) # L2 loss on spikes per neuron
            
            # Here we combine supervised loss and the regularizer
            loss_val = loss_fn(log_p_y, y_local) + reg_loss

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            local_loss.append(loss_val.item())
        
        mean_loss = np.mean(local_loss)
        loss_hist.append(mean_loss)
        live_plot(loss_hist)
        
        # Evaluate every eval_interval epochs
        if (e + 1) % eval_interval == 0 or e == nb_epochs - 1:
            train_acc = compute_classification_accuracy(x_train, y_train, train_idx)
            test_acc = compute_classification_accuracy(x_test, y_test, test_idx)
            train_acc_hist.append((e+1, train_acc))
            test_acc_hist.append((e+1, test_acc))
            
            print(f"Epoch {e+1}: loss={mean_loss:.5f}, train_acc={train_acc:.3f}, test_acc={test_acc:.3f}, lr={current_lr:.2e}")
            
            # Save best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = e + 1
                patience_counter = 0  # Reset patience counter
                
                import time
                best_model_path = f'model/best_model_dropout_epoch{e+1}_testacc{test_acc:.3f}.pth'
                save_model(
                    best_model_path,
                    w1, w2, v1,
                    loss_hist=loss_hist,
                    train_acc=train_acc,
                    test_acc=test_acc,
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


# Print configuration
print("=" * 70)
print("Training Configuration:")
print(f"  Epochs:                {nb_epochs}")
print(f"  Hidden units:          {nb_hidden}")
print(f"  Batch size:            {batch_size}")
print(f"  Learning rate:         2e-4")
print("-" * 70)
print("Dropout Configuration:")
print(f"  W1 (input-hidden):     {dropout_rate_w1:.1%}")
print(f"  V1 (hidden-hidden):    {dropout_rate_v1:.1%} (recurrent)")
print(f"  W2 (hidden-output):    {dropout_rate_w2:.1%}")
print("=" * 70)

# Train the model
loss_hist, train_acc_hist, test_acc_hist = train(
    x_train_full, y_train, train_indices,
    x_test_full, y_test, test_indices,
    lr=2e-4, 
    nb_epochs=nb_epochs,
    eval_interval=10
)

# Print accuracy history
print("\n=== Accuracy History ===")
for epoch, train_acc in train_acc_hist:
    test_acc = [acc for e, acc in test_acc_hist if e == epoch][0]
    print(f"Epoch {epoch}: Train Acc = {train_acc:.3f}, Test Acc = {test_acc:.3f}")

# Get final test batch for visualization
def get_mini_batch(x_data, y_data, valid_indices, shuffle=False):
    for ret in sparse_data_generator_from_hdf5_spikes(x_data, y_data, valid_indices, batch_size, nb_steps, nb_inputs, max_time, shuffle=shuffle):
        return ret 

x_batch, y_batch = get_mini_batch(x_test_full, y_test, test_indices)
# Evaluation mode for visualization (no dropout)
output, other_recordings = run_snn(x_batch.to_dense(), training=False)
mem_rec, spk_rec = other_recordings

# Save voltage traces
fig=plt.figure(dpi=100)
plot_voltage_traces(output)
plt.savefig('voltage_traces_dropout.png')
plt.close()

# Save spike activity
nb_plt = 4
gs = GridSpec(1,nb_plt)
fig= plt.figure(figsize=(7,3),dpi=150)
for i in range(nb_plt):
    plt.subplot(gs[i])
    plt.imshow(spk_rec[i].detach().cpu().numpy().T,cmap=plt.cm.gray_r, origin="lower" )
    if i==0:
        plt.xlabel("Time")
        plt.ylabel("Units")
    sns.despine()

plt.savefig('spike_activity_dropout.png')
plt.close()

print("\nTraining complete! Best model and visualizations saved.")