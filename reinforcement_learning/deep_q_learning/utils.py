import matplotlib
matplotlib.use("Agg")  # backend sans affichage (serveur/headless)
import matplotlib.pyplot as plt

import os
import torch
import numpy as np 
from typing import Union

from experiments.store import ExperimentPaths

#---------------------------------------- Experiment helpers ------------------------------------------------------------------------------

def save_dqn_weights(
        paths: ExperimentPaths,
        epoch: Union[int, str], 
        eval_net, 
        target_net,
        final_epoch=None
):
    """Save DQN model checkpoints for both evaluation and target networks.

    Args:
        paths (ExperimentPaths): Object containing experiment paths for saving.
        epoch (Union[int, str]): Epoch number or identifier for the checkpoint ("last" for the last model to save).
        eval_net: The evaluation Q-network
        target_net: The target Q-network
    """

    if epoch == "last":
        if final_epoch is None:
            raise ValueError("final_epoch must be provided when epoch='last'")
        epoch = final_epoch

    # Ensure the 'checkpoints' directory exists
    os.makedirs(paths.dqn_checkpoints_dir, exist_ok=True)

    # Get the specific paths from the store logic
    eval_path = paths.dqn_checkpoint_file(epoch, "eval")
    target_path = paths.dqn_checkpoint_file(epoch, "target")

    # Save the state dicts
    torch.save(eval_net.state_dict(), eval_path)
    torch.save(target_net.state_dict(), target_path)

    print(f"[INFO] Epoch {epoch} weights saved to {paths.dqn_checkpoints_dir}")

def save_dqn_training_state(
        paths: ExperimentPaths,
        epoch: int,
        eval_net,
        target_net,
        optimizer,
        replay_buffer,
        epsilon_schedule,
        delta_schedule,
        update_step: int,
        avg_losses,
        avg_len_path,
        avg_rewards,
        epsilons,
        avg_grad_norms,
        max_grad_norms,
):
    """ This function saves DQN weights and necessary elements to continue training.
    """
    os.makedirs(paths.dqn_checkpoints_dir, exist_ok=True)

    payload = {
        "epoch": epoch,
        "eval_state_dict": eval_net.state_dict(),
        "target_state_dict": target_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "update_step": update_step,

        "epsilon_schedule": {
            "init_value": epsilon_schedule.init_value,
            "value": epsilon_schedule.value,
            "decay": epsilon_schedule.decay,
            "min_value": epsilon_schedule.min_value,
        },
        "delta_schedule": {
            "init_value": delta_schedule.init_value,
            "value": delta_schedule.value,
            "decay": delta_schedule.decay,
            "min_value": delta_schedule.min_value,
        },

        "replay_buffer": {
            "current_state_memory": replay_buffer.current_state_memory,
            "next_state_memory": replay_buffer.next_state_memory,
            "action_memory": replay_buffer.action_memory,
            "reward_memory": replay_buffer.reward_memory,
            "terminated_memory": replay_buffer.terminated_memory,
            "memory_counter": replay_buffer.memory_counter,
            "idx": replay_buffer.idx,
        },

        "history": {
            "avg_losses": avg_losses,
            "avg_len_path": avg_len_path,
            "avg_rewards": avg_rewards,
            "epsilons": epsilons,
            "avg_grad_norms": avg_grad_norms,
            "max_grad_norms": max_grad_norms,
        }
    }

    torch.save(payload, paths.dqn_training_state_file)
    print(f"[INFO] Full DQN training state saved to {paths.dqn_training_state_file}")

def load_dqn_training_state(paths: ExperimentPaths, device):
    """Load DQN savings to continue training. 
    """
    checkpoint_path = paths.dqn_training_state_file

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No DQN recovery checkpoint found at {checkpoint_path}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"[INFO] Loaded DQN training state from {checkpoint_path}")
    return checkpoint

#---------------------------------------- Visualization  ------------------------------------------------------------------------------

def plot_Convergence(
        paths: ExperimentPaths, 
        avg_losses, 
        avg_len_path,
        avg_rewards,
        avg_grad_norms=None,
        max_grad_norms=None
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

    # Plot Reward
    plt.figure()
    plt.plot(avg_rewards, 'b', label='reward')
    plt.title("Convergence of DQN Algorithm reward")
    plt.xlabel("Epoch")
    plt.ylabel("Average cumulative reward")
    plt.savefig(paths.dqn_reward_plot)
    plt.close()

    # Plot average grad norm before clipping
    if avg_grad_norms is not None:
        plt.figure()
        plt.plot(avg_grad_norms, 'b', label='avg grad norm before clipping')
        plt.axhline(y=1.0, color='r', linestyle='--', label='reference = 1')
        plt.title("Average Gradient Norm Before Clipping")
        plt.xlabel("Epoch")
        plt.ylabel("Gradient norm")
        plt.legend()
        plt.savefig(paths.dqn_grad_norm_plot)
        plt.close()

    # Plot max grad norm before clipping
    if max_grad_norms is not None:
        plt.figure()
        plt.plot(max_grad_norms, 'b', label='max grad norm before clipping')
        plt.axhline(y=1.0, color='r', linestyle='--', label='reference = 1')
        plt.title("Max Gradient Norm Before Clipping")
        plt.xlabel("Epoch")
        plt.ylabel("Gradient norm")
        plt.legend()
        plt.savefig(paths.dqn_grad_norm_max_plot)
        plt.close()

#---------------------------------------- Reporting ------------------------------------------------------------------------------

def save_Data(
        paths: ExperimentPaths, 
        avg_losses, 
        avg_len_path, 
        avg_rewards,
        epsilons,
        avg_grad_norms=None,
        max_grad_norms=None
):
    os.makedirs(paths.dqn_metrics_dir, exist_ok=True)
    
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

    # Save average rewards in a CSV file
    np.savetxt(
        paths.dqn_metrics_csv("avgRewards"),
        avg_rewards,
        delimiter=",",
        header="Average Reward",
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

    if avg_grad_norms is not None:
        np.savetxt(
            paths.dqn_metrics_csv("avgGradNormBeforeClip"),
            avg_grad_norms,
            delimiter=",",
            header="Average Gradient Norm Before Clipping",
            comments=""
        )

    if max_grad_norms is not None:
        np.savetxt(
            paths.dqn_metrics_csv("maxGradNormBeforeClip"),
            max_grad_norms,
            delimiter=",",
            header="Max Gradient Norm Before Clipping",
            comments=""
        )
        
    print(f"[INFO] Losses saved to {paths.dqn_metrics_csv('losses')}")
    print(f"[INFO] Average path lengths saved to {paths.dqn_metrics_csv('avgLenPath')}")
    print(f"[INFO] Average rewards saved to {paths.dqn_metrics_csv('avgRewards')}")
    print(f"[INFO] Epsilon values saved to {paths.dqn_metrics_csv('epsilons')}")

    if avg_grad_norms is not None:
        print(f"[INFO] Avg grad norms saved to {paths.dqn_metrics_csv('avgGradNormBeforeClip')}")
    if max_grad_norms is not None:
        print(f"[INFO] Max grad norms saved to {paths.dqn_metrics_csv('maxGradNormBeforeClip')}")
        
    print(f"[INFO] DQN Metrics saved in {paths.dqn_checkpoints_dir}")
            
def save_model_complexity(
                            model,
                            params_dict, 
                            paths: ExperimentPaths
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
    
    os.makedirs(paths.dqn_metrics_dir, exist_ok=True)
    with open(paths.complexity_report_file, "w") as f:
        f.write(report)
    
    print(f"[INFO] Complexity report saved to {paths.complexity_report_file}")