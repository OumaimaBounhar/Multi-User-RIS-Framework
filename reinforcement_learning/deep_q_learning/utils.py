import os
import torch
import numpy as np 
from typing import Union
from matplotlib import pyplot as plt 
from experiments.store import ExperimentPaths

#---------------------------------------- Experiment helpers ------------------------------------------------------------------------------

def save_dqn_weights(
        paths: ExperimentPaths,
        epoch: Union[int, str], 
        eval_net, 
        target_net
):
    """Save DQN model checkpoints for both evaluation and target networks.

    Args:
        paths (ExperimentPaths): Object containing experiment paths for saving.
        epoch (Union[int, str]): Epoch number or identifier for the checkpoint ("last" for the last model to save).
        eval_net: The evaluation Q-network
        target_net: The target Q-network
    """

    # Ensure the 'checkpoints' directory exists
    os.makedirs(paths.dqn_checkpoints_dir, exist_ok=True)

    # Get the specific paths from the store logic
    eval_path = paths.dqn_checkpoint_file(epoch, "eval")
    target_path = paths.dqn_checkpoint_file(epoch, "target")

    # Save the state dicts
    torch.save(eval_net.state_dict(), eval_path)
    torch.save(target_net.state_dict(), target_path)

    # 4. Feedback
    label = f"Epoch {epoch}" if isinstance(epoch, int) else "last"
    print(f"[INFO] {label} weights saved to {paths.dqn_checkpoints_dir}")

#---------------------------------------- Visualization  ------------------------------------------------------------------------------

def plot_Convergence(
        paths: ExperimentPaths, 
        avg_losses, 
        avg_len_path
):
    # Plot Loss
    plt.figure()
    plt.plot(avg_losses, 'b', label='loss')
    plt.title("Convergence of DQN Algorithm loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average loss")
    plt.savefig(paths.dqn_loss_plot)
    plt.close()
    
    # Plot Path Length
    plt.figure()
    plt.plot(avg_len_path, 'b', label='len path')
    plt.title("Convergence of DQN Algorithm number action") 
    plt.xlabel("Epoch")
    plt.ylabel("Average len path")
    plt.savefig(paths.dqn_path_plot)
    plt.close()

#---------------------------------------- Reporting ------------------------------------------------------------------------------

def save_Data(
        paths: ExperimentPaths, 
        avg_losses, 
        avg_len_path, 
        epsilons
):
    os.makedirs(paths.dqn_checkpoints_dir, exist_ok=True)
    
    # Save loss values in a CSV file
    np.savetxt(
        paths.dqn_metrics_csv("losses"), 
        avg_losses, 
        delimiter=",", 
        header="Average Loss", 
        comments=""
    )

    # Save average len path values in a CSV file
    np.savetxt(
        paths.dqn_metrics_csv("avgLenPath"), 
        avg_len_path, 
        delimiter=",", 
        header="Average Len Path", 
        comments=""
    )

    # Save exploration rates in a CSV file
    np.savetxt(
        paths.dqn_metrics_csv("epsilons"), 
        epsilons, 
        delimiter=",", 
        header="Epsilon Values", 
        comments=""
    )

    print(f"[INFO] Losses saved to {paths.dqn_metrics_csv('losses')}")
    print(f"[INFO] Average path lengths saved to {paths.dqn_metrics_csv('avgLenPath')}")
    print(f"[INFO] Epsilon values saved to {paths.dqn_metrics_csv('epsilons')}")
    print(f"[INFO] DQN Metrics saved in {paths.dqn_checkpoints_dir}")
            
def save_model_complexity(
                            model,
                            params_dict, 
                            filename="complexity_report.txt"
):
    """
    Compute and save the model complexity and memory usage.

    """
    # Count model parameters
    n_params = sum(p.numel() for p in model.parameters())
    mem_MB = n_params * 4 / 1024**2  # FP32 weights
    mem_MB_total = mem_MB * 3        # include Adam optimizer states (approx)
    
    # Compute per-forward and per-batch ops
    input_dim = params_dict["hidden_layers"]
    hidden_layers = params_dict["params_list"]
    batch_size = params_dict["batch_size"]
    n_actions = params_dict["n_actions"]
    
    # Forward FLOPs (mult-adds)
    ops_forward = input_dim*hidden_layers[0] + sum(hidden_layers[i]*hidden_layers[i+1] for i in range(len(hidden_layers)-1)) + hidden_layers[-1]*n_actions
    
    ops_train_step = 4 * ops_forward                # forward + backward + target
    ops_per_batch = ops_train_step * batch_size
    
    # Replay buffer memory (assuming float32)
    buffer_size = params_dict.get("replay_buffer_memory_size", 80000)
    buffer_entry = (2 * input_dim + 2)  # (s, s', a, r)
    buffer_MB = buffer_size * buffer_entry * 4 / 1024**2
    
    # Write results
    report = f"""
    [MODEL COMPLEXITY REPORT]

    Network architecture: {input_dim} -> {' -> '.join(map(str, hidden_layers))} -> {n_actions}
    Total parameters: {n_params:,}
    Model memory (FP32): {mem_MB:.3f} MB
    Model + optimizer memory (≈×3): {mem_MB_total:.3f} MB
    Replay buffer: {buffer_size:,} samples × {buffer_entry} floats
    Replay buffer memory (FP32): {buffer_MB:.3f} MB

    Ops per forward pass: {ops_forward:,}
    Ops per training sample (≈4×forward): {ops_train_step:,}
    Ops per batch (batch={batch_size}): {ops_per_batch/1e6:.2f} million

    Total estimated training cost per epoch:
    ≈ n_time_steps × n_channels_train × max_len_path × ops_per_batch
    (computed analytically in paper)
    """
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(report)
    
    print(f"[INFO] Complexity report saved to {filename}")
