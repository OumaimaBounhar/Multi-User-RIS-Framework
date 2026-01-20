import torch
import os
import numpy as np 
from matplotlib import pyplot as plt 

#---------------------------------------- Experiment helpers ------------------------------------------------------------------------------

def save_model_checkpoints(
                            name,
                            epoch, 
                            eval_net, 
                            target_net
):
    checkpoint_dir =  name + '/checkpoints/'
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = checkpoint_dir + 'epoch_' + str(epoch) 
    torch.save(eval_net.state_dict(), model_path + '_eval.pth')
    torch.save(target_net.state_dict(), model_path + '_target.pth')
    print(f'Weights saved in: {model_path}')

#---------------------------------------- Visualization  ------------------------------------------------------------------------------

def plot_Convergence(
                        name, 
                        avg_losses, 
                        avg_len_path
):
    plt.plot(avg_losses, 'b', label='loss')
    #plt.plot(epsilons, 'r', label='Exploration Rate')
    plt.title("Convergence of DQN Algorithm loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average loss")
    plt.savefig(name + "/convergence_deep_q_learning.png")
    # plt.show()
    plt.close()
    
    plt.plot(avg_len_path, 'b', label='len path')
    #plt.plot(epsilons, 'r', label='Exploration Rate')
    plt.title("Convergence of DQN Algorithm number action") 
    plt.xlabel("Epoch")
    plt.ylabel("Average len path")
    plt.savefig(name + "/convergence_deep_q_learning_len_path.png")
    plt.close()

#---------------------------------------- Reporting ------------------------------------------------------------------------------

def save_Data(
                name, 
                avg_losses, 
                avg_len_path, 
                epsilons
):
    # Save loss values in a CSV file
    loss_file_path = os.path.join(name, "losses.csv")
    np.savetxt(loss_file_path, avg_losses, delimiter=",", header="Average Loss", comments="")
    
    # Save average len path values in a CSV file
    avg_len_path_file_path = os.path.join(name, "avgLenPath.csv")
    np.savetxt(avg_len_path_file_path, avg_len_path, delimiter=",", header="Average Len Path", comments="")

    # Save exploration rates in a CSV file
    epsilon_file_path = os.path.join(name, "epsilons.csv")
    np.savetxt(epsilon_file_path, epsilons, delimiter=",", header="Epsilon Values", comments="")

    print(f"[INFO] Losses saved to {loss_file_path}")
    print(f"[INFO] Epsilon values saved to {epsilon_file_path}")
            
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
