import matplotlib
matplotlib.use("Agg")  # backend sans affichage (serveur/headless)
import matplotlib.pyplot as plt

import os
import math
import torch
import numpy as np 
import pandas as pd
from tqdm import tqdm

from methods.methods import Methods
from systemModel.channel import Channel
from config.parameters import Parameters
from dataset.probability import Probability
from experiments.store import ExperimentPaths
from reinforcement_learning.deep_q_learning.agent import DeepQLearningAgent
from reinforcement_learning.q_learning.utils import load_Policy

from reinforcement_learning.deep_q_learning.components.seed import set_seed

class Test:
    """ Consolidates testing logic for Q-Learning and Deep Q-Learning. """
    
    def __init__(self, 
        parameters: Parameters, 
        methods: Methods, 
        channel: Channel, 
        probability: Probability, 
        DQN: DeepQLearningAgent
    ) -> None:
        
        self.parameters = parameters
        self.methods = methods
        self.channel = channel
        self.probability = probability
        self.DQN = DQN

    def test(
        self, 
        testing_objects_dict, 
        epoch, 
        mode='both',
        policy_network=None, 
        policy_ql=None
    ):
        """
        Core simulation engine. Saves results to disk and generates plots

        Args:
        @testing_objects_dict: dict containing all necessary objects for testing (channel, feedback, etc.)
        @epoch: current epoch number for logging
        @mode: 'dqn' (tests baselines + DQN), 'ql' (
        tests baselines + QL), or 'both' (tests all).
        @policy_network: the DQN policy network to use for testing (if mode includes DQN)
        @policy_ql: the Q-Learning policy to use for testing (if mode includes QL)
        """
        n_ex = 500 # Number of times we simulate the channel for each SNR
        T = 10     # Time steps for the evolution of the channel
        
        objs = testing_objects_dict
        filename = objs["filename"]

        # Methods to compare
        # Define labels based on the selected mode
        baselines = ["Exhaustive", "Hierarchical", "Random sampling"]
        if mode == 'dqn':
            label = baselines + ["Deep Q-Learning"]
        elif mode == 'ql':
            label = baselines + ["Q-Learning"]
        else: # both
            label = baselines + ["Deep Q-Learning", "Q-Learning"]

        # Loop over all SNR values
        for SNR in objs["snr_values"]:
            # Ensure channel noise parameters is updated based on each SNR
            self.parameters.set_SNR(SNR)
            self.parameters.std_noise = math.sqrt(10**(-SNR/10))
            self.parameters.set_noise(mean_noise=0, std_noise=self.parameters.std_noise)
            
            print(f'SNR value = {self.parameters.SNR}')
            print(f'Std noise = {self.parameters.std_noise}')
            print(f'Noise parameters = {self.parameters.get_noise_parameters}')

            # Initialize Methods with the specific policy/network for this epoch
            methods = Methods(
                objs["parameters"],
                objs["channel"], 
                objs["feedback"], 
                objs["probability"], 
                objs["states"], 
                policy_ql if policy_ql else objs["Policy_Q"], 
                policy_network
            )
            
            correct_class = np.zeros((T, len(label)))
            relative_strength = np.zeros((T, len(label)))
            all_len_path_per_SNR = [] # List to store lengths of paths during testing per SNR
            successful_episodes_per_epoch = [] # List to store number of successful episodes per epoch
            total_successful_episodes = 0 # To count successful simulations
            
            print(f'[INFO] Testing Mode: {mode} | SNR: {SNR} | Epoch: {epoch} | Std Noise: {self.parameters.std_noise}')
            
            set_seed(1000 + SNR + epoch)

            for _ in tqdm(range(n_ex)):
                objs["channel"].new_channel() # New channel
                methods.forget() # We reinitialize the different algorithms
                successful_episode = False # To check if there is at least one successful episode in this simulation
                total_len_path = T
                
                for t in range(T):
                    opt_idx_cd = methods.optimal_codeword() # Gives the index of the best codeword for the current channel realization (without noise)
                    objs["feedback"].transmit(opt_idx_cd, codebook_used=0)
                    RSE_opt = objs["feedback"].get_feedback(noise=False)
            
                    # Collect indices based on mode
                    indices = [
                        methods.exhaustive(), 
                        methods.hierarchical(), 
                        methods.random_sampling()
                        ]
                    
                    if mode == 'dqn':
                        indices.append(methods.deep_q_learning_sampling())

                    elif mode == 'ql':
                        indices.append(methods.q_learning_sampling())

                    else: # both
                        indices.extend([methods.deep_q_learning_sampling(), methods.q_learning_sampling()])
                    
                    # Calculate metrics to measure performance of each method
                    correct_class[t] += [idx == opt_idx_cd for idx in indices]
                    
                    ## Later proposition --- Modify to compute relative strength for each method compared to the optimal codeword
                    # correct_class[t] = correct_class[t] + [feedback.transmit(index) == index_optimal_cd for index in list_algorithms_class]
                    
                    for i, idx in enumerate(indices):
                        objs["feedback"].transmit(idx, codebook_used=0)
                        rse = objs["feedback"].get_feedback(noise=False)
                        relative_strength[t][i] += (rse / RSE_opt)
                    
                    # Track success (using the relevant agent for the current test)
                    # Track success for the "primary" agent in the test
                    # E.g., for 'both', it tracks DQN (index 3) and QL (index 4)
                    success_check_idx = 3
                    if indices[success_check_idx] == opt_idx_cd and not successful_episode:
                        successful_episode = True
                        total_len_path = t
                        
                    objs["channel"].update(modification_channel=objs["modification_channel"])
                
                if successful_episode: 
                    total_successful_episodes += 1
                
                successful_episodes_per_epoch.append(total_successful_episodes)
                all_len_path_per_SNR.append(total_len_path)

            # Averaging
            avg_len_path = np.mean(all_len_path_per_SNR)
            correct_class /= n_ex
            relative_strength /= n_ex
            
            # Process and Save Statistics
            self._save_and_plot(
                filename, 
                epoch, 
                SNR, 
                label, 
                T, 
                correct_class, 
                relative_strength, 
                successful_episodes_per_epoch, 
                avg_len_path, 
                total_successful_episodes, 
                n_ex
            )

    def _save_and_plot(
        self, 
        filename, 
        epoch, 
        SNR, 
        label, 
        T, 
        correct_class, 
        relative_strength, 
        successful_episodes_per_epoch, 
        avg_len_path, 
        total_successful_episodes, 
        n_ex
    ):
        """ Internal helper to keep the main loop clean. 
        """
        
        # Save .dat files
        pd.DataFrame({"Epoch": epoch, "AvgLenPath": avg_len_path}, index=[0]).to_csv(
            os.path.join(filename, f"data_epoch_{epoch}_snr_{SNR}.dat"), index=False)
        pd.DataFrame(correct_class, columns=label).to_csv(
            os.path.join(filename, f"probability_epoch_{epoch}_snr_{SNR}.dat"), index=False)
        pd.DataFrame(relative_strength, columns=label).to_csv(
            os.path.join(filename, f"strength_epoch_{epoch}_snr_{SNR}.dat"), index=False)

        # 1. Probability Plot
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, T + 1), correct_class, label=label)
        plt.legend()
        plt.title(f'Probability of Finding the Best Codeword for Different Number of Pilots Sent at SNR = {SNR}')
        plt.grid()
        plt.savefig(f"{filename}/Image_snr_{SNR}_epoch_{epoch}.png")
        plt.close()
        
        # 2. Strength Plot
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, T + 1), relative_strength, label=label)
        plt.legend()
        plt.title(f'Strength for Different Number of Pilots Sent (SNR = {SNR})')
        plt.grid()
        plt.savefig(f"{filename}/Image_strength_epoch_{epoch}_snr_{SNR}.png")
        plt.close()

        # 3. Successful Episodes Plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(successful_episodes_per_epoch) + 1), successful_episodes_per_epoch, color='b')
        plt.xlabel('Epoch')
        plt.ylabel('Number of Successful Episodes')
        plt.title(f'Number of Successful Episodes by Epoch at SNR = {SNR}')
        plt.grid(True)
        plt.savefig(f"{filename}/successful_episodes_by_epoch_at_snr_{SNR}.png")
        plt.close()

        print(f"[INFO] SNR {SNR}: Success Rate {(total_successful_episodes/n_ex)*100}% | Avg Path {avg_len_path}")

    def run_model_tests(
        self, 
        testing_objects_dict, 
        checkpoints_dir_ql, 
        checkpoints_dir_dql, 
        mode='both'
    ):
        """
        Runner to save parameters for QL and DQN and loop through checkpoints.
    
        Args:
        @testing_objects_dict: dict containing all necessary objects for testing (channel, feedback, etc.)
        @checkpoints_dir_ql: directory where QL policies are stored
        @checkpoints_dir_dql: directory where DQN checkpoints are stored
        @mode: 'dqn' (tests baselines + DQN), 'ql' (tests baselines + QL), or 'both' (tests all).   
        """
        # Save params back in Parameters class
        self.parameters.save_to_file(
            os.path.join(
                testing_objects_dict["filename"], 
                f"params_{mode}.txt"
            ),
            mode
        )

        paths = ExperimentPaths(root=testing_objects_dict["filename"])

        # Logic to find checkpoints based on mode
        if mode in ('both', 'dqn'):
            dqn_files = [
                f for f in os.listdir(checkpoints_dir_dql) 
                if f.endswith('_eval.pth')
            ]

            # sSort name of files by epoch number
            dqn_files.sort(key=lambda x: int(x.split('_')[1]))

            for f in dqn_files:

                # Extract epoch number from filename (assuming format 'checkpoint_epoch_eval.pth')
                epoch = int(f.split('_')[1])
                
                # Reseed to ensure reproducible test conditions
                set_seed(42 + epoch)

                if epoch % self.parameters.test_freq != 0:
                    continue

                if self.DQN is None:
                    raise ValueError("DQN agent is required to run DQN checkpoint tests.")

                device = next(self.DQN.evaluation_q_network.parameters()).device

                checkpoint_path = os.path.join(checkpoints_dir_dql, f)

                state_dict = torch.load(checkpoint_path, map_location=device)

                self.DQN.evaluation_q_network.load_state_dict(state_dict)

                self.DQN.evaluation_q_network.to(device)
                self.DQN.evaluation_q_network.eval()

                policy_ql = None

                if mode == 'both':
                    # Attempt to find matching QL policy by epoch
                    try:
                        policy_ql = load_Policy(
                            paths= paths,
                            episode = epoch
                        )
                        
                    except FileNotFoundError:
                        policy_ql = None
                        
                self.test(
                    testing_objects_dict, 
                    epoch, 
                    mode=mode, 
                    policy_network=self.DQN.evaluation_q_network, 
                    policy_ql=policy_ql
                )
                    
        elif mode == 'ql':

            ql_files = [
                f for f in os.listdir(checkpoints_dir_ql) 
                if f.startswith('policy_after_') 
                and f.endswith('.csv')
            ]

            # Sort name of files by episode (epoch) number (assuming format 'policy_after_X_episodes.csv')
            ql_files.sort(key=lambda x: int(x.split('_')[2].replace('episodes.csv','')))

            for f in ql_files:

                epoch = int(f.split('_')[2].replace('episodes.csv',''))

                if epoch % self.parameters.test_freq == 0:

                    policy_ql = load_Policy(
                        paths= paths,
                        episode = epoch
                    )

                    self.test(
                        testing_objects_dict, 
                        epoch, 
                        mode='ql', 
                        policy_ql=policy_ql
                    )
