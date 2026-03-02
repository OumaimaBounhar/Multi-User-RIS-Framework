import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

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

def plot_Convergence(name, smoothed_avg_len):
    """Plot the convergence of the Q-Learning Algorithm

    Args:
        name (_type_): _description_
        smoothed_avg_len (_type_): _description_
    """
    plt.plot(smoothed_avg_len)
    plt.title("Convergence of Q-Learning Algorithm")
    plt.xlabel("Iteration")
    plt.ylabel("Average Len path normalized with a window size = 10")
    plt.savefig(name + "/convergence_q_learning_train.png")

#---------------------------------------- Reporting ------------------------------------------------------------------------------

def save_Q_matrix(Q_matrix, episode: int,name:str) -> None:
    """" Saves the Q-Learning matrix in a csv file
    Args : 
        episode : The index of the episode
    Returns :
        A csv file
    """
    # Ensure the directory exists
    os.makedirs(name, exist_ok=True)
    
    # Save the Q-matrix as a CSV file
    filename = name + f"/Q_matrix_after_{episode}episodes.csv"
    np.savetxt(filename, Q_matrix, delimiter=",", fmt='%.6f')
    
    print(f"[INFO] Q-matrix saved to {filename}")
    
def save_policy(policy, episode: int,name:str) -> None:
    """" Saves the Q-Learning matrix in a csv file
    Args : 
        episode : The index of the episode
    Returns :
        A csv file
    """
    # Save the policy matrix to a CSV file
    filename_policy = name + f"/policy_after_{episode}episodes.csv"
    np.savetxt(filename_policy, policy, delimiter=",", fmt='%d')
    print(f"[INFO] Policy saved to {filename_policy }")

def save_frequency_update_per_state(Q_matrix_freq, name, n_episodes, delta_init, learning_rate_init) -> None:
    """
    Save the percentage of visits per state.
    """

    # Avoid division by zero
    total_updates = max(np.sum(Q_matrix_freq), 1)
    frequency_percent = Q_matrix_freq / total_updates * 100

    filename = (
        f"{name}/frequency_after_{n_episodes}"
        f"_episodes_delta_{delta_init}"
        f"_alpha_{learning_rate_init}.csv"
    )

    np.savetxt(filename, frequency_percent, delimiter=",")

    print(f"[INFO] Frequency matrix saved to {filename}")

def save_training_metrics(name, avg_len_train_epoch: list) -> None:
    """
    Save training metrics (average path length per episode).
    """

    output_df = pd.DataFrame({
        "Len Train": avg_len_train_epoch
    })

    filename = f"{name}/Data_len_path_training.dat"
    output_df.to_csv(filename, index=False)

    print(f"[INFO] Mean len path during QL - Training saved to {filename}")

def load_Q_matrix(n_episodes: int,name:str) -> List[List[float]]:
    """Loads the Q-matrix from a CSV file and returns it.
    
    Args:
        n_episodes (int): Number of episodes.
    
    Returns:
        list: List of lists representing the Q-matrix.
    """
    filename = name+f"/Q_matrix_after_{n_episodes}episodes.csv"
    
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        Q_matrix = [[float(item) for item in row] for row in reader]
    return Q_matrix

def load_Policy(n_episodes: int,name:str):
    """Loads the policy matrix from a CSV file.

    Args:
        n_episodes (int): Number of episodes.
    """
    # Define the filename
    filename = name+f"/policy_after_{n_episodes}episodes.csv"

    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        #Policy = [[int(item) for item in row] for row in reader]
        Policy = [int(row[0]) for row in reader]
    return Policy
