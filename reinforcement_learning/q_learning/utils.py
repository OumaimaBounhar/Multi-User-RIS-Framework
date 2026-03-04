import os, csv
import numpy as np
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
from experiments.store import ExperimentPaths

#---------------------------------------- Experiment helpers ------------------------------------------------------------------------------

def extract_Policy(Q_matrix) -> np.ndarray:
    """Greedy Policy extraction from the current Q-matrix.

    Args:
        Q_matrix (np.ndarray): shape (n_states, n_actions)

    Returns:
        policy (np.ndarray): shape (n_states,)
    """
    n_states = Q_matrix.shape[0]
    policy = np.zeros(n_states, dtype = int)

    for s in range(n_states):
        q_value = Q_matrix[s]
        max_q_value = np.max(q_value)
        best_actions = np.flatnonzero(np.isclose(q_value, max_q_value))
        policy[s] = int(np.random.choice(best_actions))

    return policy

def computes_percentage_unvisited_states(Q_matrix_freq) -> float:
    """Computes the percentage of unvisited states in the Q-matrix.
    Args:
        Q_matrix_freq: The matrix that tracks the states that were visited
    
    Returns:
        percentage_unvisited_states (float): The percentage of unvisited states in the Q-matrix.
    """
    n_states = Q_matrix_freq.shape[0]
    assert n_states > 0

    unvisited_states = np.sum(Q_matrix_freq == 0)
    percentage_unvisited_states = 100.0 * unvisited_states / n_states

    return percentage_unvisited_states

#---------------------------------------- Visualization  ------------------------------------------------------------------------------

def plot_Convergence(paths: ExperimentPaths, smoothed_avg_len):
    """Plot the convergence of the Q-Learning Algorithm

    Args:
        paths (ExperimentPaths): The paths of the experiment
        smoothed_avg_len (nd_array): The smoothed average length of the path during training
    """
    plt.figure()
    plt.plot(smoothed_avg_len)
    plt.title("Convergence of Q-Learning Algorithm")
    plt.xlabel("Iteration")
    plt.ylabel("Average Len path normalized (window=10)")
    
    filename = paths.ql_convergence_plot_file
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()

#---------------------------------------- Reporting ------------------------------------------------------------------------------

def save_Q_matrix(paths: ExperimentPaths, Q_matrix, episode: int, is_last: bool = False) -> None:
    """" Saves the Q-Learning matrix in a csv file
    Args : 
        paths (ExperimentPaths): The paths of the experiment
        Q_matrix (np.ndarray): The Q-matrix to save
        episode (int): The index of the episode
        is_last (bool): Whether this is the final Q-matrix after training completion. If True, the filename will indicate it's the last one.
    """
    # Ensure the directory exists
    filename = paths.q_matrix_file(episode, is_last)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savetxt(filename, Q_matrix, delimiter=",", fmt='%.6f')
    print(f"[INFO] Q-matrix saved to {filename}")
    
def save_Policy_matrix(paths: ExperimentPaths, policy, episode: int, is_last: bool = False) -> None:
    """" Saves the Policy matrix in a csv file
    Args : 
        paths (ExperimentPaths): The paths of the experiment
        policy (np.ndarray): The policy to save
        episode (int): The index of the episode
        is_last (bool): Whether this is the final policy matrix after training completion. If True, the filename will indicate it's the last one.
    """
    # Save the policy matrix to a CSV file
    filename_policy = paths.policy_matrix_file(episode, is_last)
    os.makedirs(os.path.dirname(filename_policy), exist_ok=True)
    np.savetxt(filename_policy, policy, delimiter=",", fmt='%d')
    print(f"[INFO] Policy saved to {filename_policy}")

def save_frequency_update_per_state(paths: ExperimentPaths, Q_matrix_freq, n_episodes, delta_init, learning_rate_init) -> None:
    """
    Save the percentage of visits per state.
    """
    # Avoid division by zero
    total_updates = max(np.sum(Q_matrix_freq), 1)
    frequency_percent = Q_matrix_freq / total_updates * 100

    filename = paths.ql_frequency_matrix_file(n_episodes, delta_init, learning_rate_init)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savetxt(filename, frequency_percent, delimiter=",")
    print(f"[INFO] Frequency matrix saved to {filename}")

def save_training_metrics(paths: ExperimentPaths, avg_len_train_epoch: list) -> None:
    """
    Save training metrics (average path length per episode).
    """

    output_df = pd.DataFrame({
        "Len Train": avg_len_train_epoch
    })

    filename = paths.ql_training_metrics_file
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    output_df.to_csv(filename, index=False)
    print(f"[INFO] Mean len path during QL - Training saved to {filename}")

def load_Q_matrix(paths: ExperimentPaths, episode: int) -> List[List[float]]:
    """Loads the Q-matrix from a CSV file and returns it.
    
    Args:
        paths (ExperimentPaths): The paths of the experiment
        episode (int): The episode number.
    
    Returns:
        list: List of lists representing the Q-matrix.
    """
    filename = paths.q_matrix_file(episode)
    
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        Q_matrix = [[float(item) for item in row] for row in reader]
    return Q_matrix

def load_Policy(paths: ExperimentPaths, episode: int, is_last: bool = False) -> List[int]:
    """Loads the policy matrix from a CSV file.

    Args:
        paths (ExperimentPaths): The paths of the experiment
        episode (int): The episode number.
        is_last (bool): Whether this is the final policy matrix after training completion. If True, the filename will indicate it's the last one.

    Returns:
        Policy (List[int]): List of integers representing the policy.
    """
    # Define the filename
    filename = paths.policy_matrix_file(episode, is_last=is_last)

    if not os.path.exists(filename):
        raise FileNotFoundError(f"No policy file found at {filename}. Please ensure the file exists or check the episode number and is_last flag.")
    
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        Policy = [int(row[0]) for row in reader]
    print(f"[INFO] Successfully loaded policy from {filename}")
    return Policy
