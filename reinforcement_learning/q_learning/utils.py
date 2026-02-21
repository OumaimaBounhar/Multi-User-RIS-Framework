import os
import csv
import numpy as np
import matplotlib.pyplot as plt

#---------------------------------------- Experiment helpers ------------------------------------------------------------------------------

def computes_percentage_unvisited_states() -> float:
    """Computes the percentage of unvisited states in the Q-matrix.

    Returns:
        percentage_unvisited_states (float): The percentage of unvisited states in the Q-matrix.
    """
    # compute the number of unvisited states
    number_of_unvisited_states = np.count_nonzero(self.Q_matrix == self.initial_value_Q_matrix)
    # compute the percentage of unvisited states
    percentage_unvisited_states = 100 * number_of_unvisited_states / self.Q_matrix.size
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

def save_Q_matrix(episode: int,name:str) -> None:
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
    np.savetxt(filename, self.Q_matrix, delimiter=",", fmt='%.6f')
    
    print(f"[INFO] Q-matrix saved to {filename}")
    
def save_policy(episode: int,name:str) -> None:
    """" Saves the Q-Learning matrix in a csv file
    Args : 
        episode : The index of the episode
    Returns :
        A csv file
    """
    # Save the policy matrix to a CSV file
    filename_policy = name + f"/policy_after_{episode}episodes.csv"
    np.savetxt(filename_policy, self.policy, delimiter=",", fmt='%d')
    print(f"[INFO] Policy saved to {filename_policy }")

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