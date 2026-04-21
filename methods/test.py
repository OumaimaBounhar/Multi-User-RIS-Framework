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
    
    def _build_method_specs(
        self, 
        methods, 
        mode
    ):
        """
        Returns a list of dicts:
        {
            "label": str,
            "run": callable,
            "track_success": bool
        }
        """
        _, codebook_specs = self.parameters.get_codebook_parameters()
        pilot_kind = codebook_specs[1].kind.lower()

        specs = []

        # Always available
        specs.append({
            "label": "Exhaustive",
            "run": methods.exhaustive,
            "track_success": False,
        })

        # Only include hierarchical search when the pilot codebook is hierarchical
        if pilot_kind == "hierarchical":
            specs.append({
                "label": "Hierarchical",
                "run": lambda: methods.hierarchical(
                    noisy_measurement=self.parameters.hierarchical_noisy_measurement
                ),
                "track_success": False,
            })

        if mode == "dqn":
            specs.append({
                "label": "Deep Q-Learning",
                "run": methods.deep_q_learning_sampling,
                "track_success": True,
            })

        elif mode == "ql":
            specs.append({
                "label": "Q-Learning",
                "run": methods.q_learning_sampling,
                "track_success": True,
            })
            specs.append({
            "label": "Random sampling",
            "run": methods.random_sampling,
            "track_success": False,
            })

        else:  # both
            specs.append({
                "label": "Deep Q-Learning",
                "run": methods.deep_q_learning_sampling,
                "track_success": True,
            })
            specs.append({
                "label": "Q-Learning",
                "run": methods.q_learning_sampling,
                "track_success": False,
            })
            specs.append({
            "label": "Random sampling",
            "run": methods.random_sampling,
            "track_success": False,
            })

        return specs

    def _epoch_already_tested(
        self,
        paths,
        epoch,
        snr_values
    ):
        missing_files = []

        for snr in snr_values:
            expected_files = [
                paths.test_summary_file(epoch, snr),
                paths.test_probability_file(epoch, snr),
                paths.test_strength_file(epoch, snr),
                paths.test_strength_success_file(epoch, snr),
                paths.test_strength_failure_file(epoch, snr),
                paths.test_probability_plot(epoch, snr),
                paths.test_strength_plot(epoch, snr),
                paths.test_strength_success_plot(epoch, snr),
                paths.test_strength_failure_plot(epoch, snr),
                paths.test_success_plot(epoch, snr),
            ]

            missing_for_snr = [f for f in expected_files if not os.path.exists(f)]

            if missing_for_snr:
                print(f"[TEST] epoch {epoch}, snr {snr}: missing {len(missing_for_snr)} files")
                for f in missing_for_snr:
                    print(f"[TEST] missing: {f}")

                missing_files.extend(missing_for_snr)

        if missing_files:
            return False

        return True

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

        Metrics computed:
        - correct_class: probability of selecting the true optimal codeword
        - normalized_power_all: average (RSE_selected / RSE_opt) over all runs
        - normalized_power_on_success: average (RSE_selected / RSE_opt) only on successful runs
        """
        print(f"[TEST] ENTER test(): epoch={epoch}, mode={mode}")
        n_ex = 500 # Number of times we simulate the channel for each SNR
        T = 8     # Time steps for the evolution of the channel
        
        objs = testing_objects_dict
        paths = objs["paths"]

        # Loop over all SNR values
        for SNR in objs["snr_values"]:
            # Ensure channel noise parameters is updated based on each SNR
            self.parameters.set_SNR(SNR)
            self.parameters.std_noise = math.sqrt(10**(-SNR/10))
            self.parameters.set_noise(mean_noise=0, std_noise=self.parameters.std_noise)
            
            print(f'SNR value = {self.parameters.SNR}')
            print(f'Std noise = {self.parameters.std_noise}')
            print(f'Noise parameters = {self.parameters.get_noise_parameters()}')

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
            
            # Define labels based on the selected mode
            method_specs = self._build_method_specs(methods, mode)
            label = [spec["label"] for spec in method_specs]
            n_methods = len(label)

            primary_method_idx = next(
                (i for i, spec in enumerate(method_specs) if spec["track_success"]),
                None
            )

            correct_class = np.zeros((T, n_methods), dtype=float)

            # Average normalized power over ALL runs
            normalized_power_all = np.zeros((T, n_methods), dtype=float)

            # Sum of normalized power only when success happens
            normalized_power_success_sum = np.zeros((T, n_methods), dtype=float)
            # Number of successes per time t and per method
            success_counts_per_t = np.zeros((T, n_methods), dtype=float)

            # Sum of normalized power only when failure happens
            normalized_power_failure_sum = np.zeros((T, n_methods), dtype=float)
            # Number of failures per time t and per method
            failure_counts_per_t = np.zeros((T, n_methods), dtype=float)
            
            # List to store number of successful episodes per epoch
            successful_episodes_per_epoch = [] 
            total_successful_episodes = 0 # To count successful simulations

            # List to store lengths of paths during testing per SNR
            all_len_path_per_SNR = [] 
            
            print(f'[INFO] Testing Mode: {mode} | SNR: {SNR} | Epoch: {epoch} | Std Noise: {self.parameters.std_noise}')
            
            # Reseed to ensure reproducible test conditions
            set_seed(42)

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
                    indices = [spec["run"]() for spec in method_specs]
                                    
                    # Calculate metrics to measure performance of each method
                    for i, idx in enumerate(indices):
                        is_success = (idx == opt_idx_cd)

                        if is_success:
                            correct_class[t, i] += 1.0
                            success_counts_per_t[t, i] += 1.0
                        else:
                            failure_counts_per_t[t, i] += 1.0

                        objs["feedback"].transmit(idx, codebook_used=0)
                        rse_selected  = objs["feedback"].get_feedback(noise=False)

                        normalized_ratio = rse_selected / RSE_opt if RSE_opt > 0 else 0.0

                        if is_success and not np.isclose(normalized_ratio, 1.0, atol=1e-8):
                            print(
                                f"[DEBUG] success but normalized_ratio={normalized_ratio:.12f}, "
                                f"idx={idx}, opt_idx_cd={opt_idx_cd}"
                            )

                        # Unconditional average power ratio
                        normalized_power_all[t, i] += normalized_ratio

                        # Conditional average power ratio on successful detections only
                        if is_success:
                            normalized_power_success_sum[t, i] += normalized_ratio
                        else:
                            normalized_power_failure_sum[t, i] += normalized_ratio
                    
                    # Track success (using the relevant agent for the current test)
                    if (
                    primary_method_idx is not None
                    and indices[primary_method_idx] == opt_idx_cd
                    and not successful_episode
                    ):
                        successful_episode = True
                        total_len_path = t
                        
                    objs["channel"].update(modification_channel=objs["modification_channel"])
                
                if successful_episode: 
                    total_successful_episodes += 1
                
                successful_episodes_per_epoch.append(total_successful_episodes)
                all_len_path_per_SNR.append(total_len_path)

            # Averaging
            avg_len_path = np.mean(all_len_path_per_SNR)
            # Convert counts to probabilities / averages
            correct_class /= n_ex
            normalized_power_all /= n_ex

            normalized_power_on_success = np.divide(
                normalized_power_success_sum,
                success_counts_per_t,
                out=np.full_like(normalized_power_success_sum, np.nan),
                where=success_counts_per_t > 0
            )
            
            normalized_power_on_failure = np.divide(
                normalized_power_failure_sum,
                failure_counts_per_t,
                out=np.full_like(normalized_power_failure_sum, np.nan),
                where=failure_counts_per_t > 0
            )
            # Process and Save Statistics
            self._save_and_plot(
                paths, 
                epoch, 
                SNR, 
                label, 
                T, 
                correct_class, 
                normalized_power_all,
                normalized_power_on_success,
                normalized_power_on_failure,
                successful_episodes_per_epoch, 
                avg_len_path, 
                total_successful_episodes, 
                n_ex
            )

    def _save_and_plot(
        self, 
        paths, 
        epoch, 
        SNR, 
        label, 
        T, 
        correct_class, 
        normalized_power_all, 
        normalized_power_on_success, 
        normalized_power_on_failure,
        successful_episodes_per_epoch, 
        avg_len_path, 
        total_successful_episodes, 
        n_ex
    ):
        """ Internal helper to save statistics and generate plots.
        """
        print(f"[TEST] ENTER _save_and_plot(): epoch={epoch}, SNR={SNR}")

        size_codebooks, codebook_specs = self.parameters.get_codebook_parameters()
        comm_kind = codebook_specs[0].kind
        pilot_kind = codebook_specs[1].kind

        comm_size = size_codebooks[0]
        pilot_size = size_codebooks[1]

        # Save .dat files
        pd.DataFrame(
            {"Epoch": epoch, "AvgLenPath": avg_len_path}, 
            index=[0]).to_csv(
            paths.test_summary_file(epoch, SNR), 
            index=False
        )

        pd.DataFrame(
            correct_class, 
            columns=label).to_csv(
            paths.test_probability_file(epoch, SNR), 
            index=False
        )

        pd.DataFrame(
            normalized_power_all, 
            columns=label
        ).to_csv(
            paths.test_strength_file(epoch, SNR), 
            index=False
        )

        # Conditional normalized power on success
        pd.DataFrame(
            normalized_power_on_success,
            columns=label
        ).to_csv(
            paths.test_strength_success_file(epoch, SNR),
            index=False
        )

        # Conditional normalized power on failure
        pd.DataFrame(
            normalized_power_on_failure,
            columns=label
        ).to_csv(
            paths.test_strength_failure_file(epoch, SNR),
            index=False
        )

        # 1. Success probability plot
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, T + 1), correct_class, label=label)
        plt.legend()
        plt.title(
            f'Success Probability of Finding the Best Codeword vs Number of Pilots\n'
            f'Epoch={epoch} | SNR={SNR} dB | Communication: {comm_kind} ({comm_size}) | Pilots: {pilot_kind} ({pilot_size})'
        )
        plt.xlabel('Number of pilots')
        plt.ylabel('Success probability')
        plt.grid()
        plt.savefig(paths.test_probability_plot(epoch, SNR))
        plt.close()

        # 2. Unconditional normalized power plot
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, T + 1), normalized_power_all, label=label)
        plt.legend()
        plt.title(
            f'Average Normalized Power (All Runs)\n'
            f'Epoch={epoch} | SNR={SNR} dB | Communication: {comm_kind} ({comm_size}) | Pilots: {pilot_kind} ({pilot_size})'
        )
        plt.xlabel('Number of pilots')
        plt.ylabel(r'$P(\hat{\phi}) / P(\phi^\star)$')
        plt.grid()
        plt.savefig(paths.test_strength_plot(epoch, SNR))
        plt.close()

        # # 3. Conditional normalized power plot on success
        # plt.figure(figsize=(10, 6))
        # plt.plot(np.arange(1, T + 1), normalized_power_on_success, label=label)
        # plt.legend()
        # plt.title(
        #     f'Average Normalized Power Conditioned on Success\n'
        #     f'Epoch={epoch} | SNR={SNR} dB | Communication: {comm_kind} ({comm_size}) | Pilots: {pilot_kind} ({pilot_size})'
        # )
        # plt.xlabel('Number of pilots')
        # plt.ylabel(r'$P(\hat{\phi}) / P(\phi^\star)$ | success')
        # plt.grid()
        # plt.savefig(paths.test_strength_success_plot(epoch, SNR))
        # plt.close()

        # # 5. Conditional normalized power plot on failure
        # plt.figure(figsize=(10, 6))
        # plt.plot(np.arange(1, T + 1), normalized_power_on_failure, label=label)
        # plt.legend()
        # plt.title(
        #     f'Average Normalized Power Conditioned on Failure\n'
        #     f'Epoch={epoch} | SNR={SNR} dB | Communication: {comm_kind} ({comm_size}) | Pilots: {pilot_kind} ({pilot_size})'
        # )
        # plt.xlabel('Number of pilots')
        # plt.ylabel(r'$P(\hat{\phi}) / P(\phi^\star)$ | failure')
        # plt.grid()
        # plt.savefig(paths.test_strength_failure_plot(epoch, SNR))
        # plt.close()

        # 6. Successful episodes plot
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, len(successful_episodes_per_epoch) + 1),
            successful_episodes_per_epoch,
            color='b'
        )
        plt.xlabel('Simulation index')
        plt.ylabel('Cumulative successful episodes')
        plt.title(
            f'Number of Successful Episodes\n'
            f'Epoch={epoch} | SNR={SNR} dB | Communication: {comm_kind} ({comm_size}) | Pilots: {pilot_kind} ({pilot_size})'
        )
        plt.grid(True)
        plt.savefig(paths.test_success_plot(epoch, SNR))
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
        paths = testing_objects_dict["paths"]

        # Logic to find checkpoints based on mode
        if mode in ('both', 'dqn'):
            dqn_files = [
                f for f in os.listdir(checkpoints_dir_dql) 
                if f.endswith('_eval.pth')
            ]

            print(f"[TEST] Found {len(dqn_files)} DQN checkpoints in {checkpoints_dir_dql}")

            # Sort name of files by epoch number
            dqn_files.sort(key=lambda x: int(x.split('_')[1]))

            for f in dqn_files:
                # Extract epoch number from filename (assuming format 'checkpoint_epoch_eval.pth')
                epoch = int(f.split('_')[1])

                print(f"[TEST] considering checkpoint file = {f}")
                print(f"[TEST] extracted epoch = {epoch}")
                
                if epoch % self.parameters.test_freq != 0:
                    continue

                print(f"[TEST] checking if epoch {epoch} already tested")
                if self._epoch_already_tested(
                    paths, 
                    epoch, 
                    testing_objects_dict["snr_values"]
                ):
                    continue

                if self.DQN is None:
                    raise ValueError("DQN agent is required to run DQN checkpoint tests.")

                device = next(self.DQN.evaluation_q_network.parameters()).device
                checkpoint_path = os.path.join(checkpoints_dir_dql, f)
                print(f"[TEST] loading checkpoint: {checkpoint_path}")

                try:
                    state_dict = torch.load(checkpoint_path, map_location=device)
                    self.DQN.evaluation_q_network.load_state_dict(state_dict)

                except Exception as e:
                    print(f"[ASYNC-EVAL] Skipping epoch {epoch} for now, checkpoint not ready: {e}")
                    continue

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
                
                print(f"[TEST] launching test() for epoch {epoch}")
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

            print(f"[TEST] Found {len(ql_files)} QL checkpoints in {checkpoints_dir_ql}")

            # Sort name of files by episode (epoch) number (assuming format 'policy_after_X_episodes.csv')
            ql_files.sort(key=lambda x: int(x.split('_')[2].replace('episodes.csv','')))

            for f in ql_files:

                epoch = int(f.split('_')[2].replace('episodes.csv',''))

                print(f"[TEST] considering checkpoint file = {f}")
                print(f"[TEST] extracted epoch = {epoch}")

                if epoch % self.parameters.test_freq == 0:

                    policy_ql = load_Policy(
                        paths= paths,
                        episode = epoch
                    )

                    print(f"[TEST] launching test() for epoch {epoch}")
                    self.test(
                        testing_objects_dict, 
                        epoch, 
                        mode='ql', 
                        policy_ql=policy_ql
                    )
