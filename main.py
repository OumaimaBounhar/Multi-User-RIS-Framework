import numpy as np

from config.parameters import Parameters
from systemModel.codebooks import CodebookSpec, Codebooks
from systemModel.channel import Channel
from systemModel.feedback import Feedback
from systemModel.signal import Signal

from experiments.store import Store, ExperimentPaths
from experiments.dataset_factory import DatasetFactory, DatasetMode
from experiments.noise_factory import NoiseFactory, NoiseMode

from dataset.probability import Probability
from reinforcement_learning.states import State
from reinforcement_learning.env import Environment
from reinforcement_learning.deep_q_learning.components.seed import set_seed
from experiments.runner import Runner

def main():
    # Reproducibility
    set_seed(42)

    all_size_of_codebooks = [
                                [8, 14],
                                # [16, 30],
                                # [32, 62],
                                # [64, 126]
                            ]

    all_codebook_specs = [
                            [CodebookSpec(kind="narrow", N=8),
                            CodebookSpec(kind="hierarchical", K=3, M=2)],
                            # [CodebookSpec(kind="narrow", N=16),
                            #  CodebookSpec(kind="hierarchical", K=4, M=2)],
                        ]

    delta_values = [
                    # 3e-1,
                    # 2e-1,
                    1e-2,
                    # 2e-2
                    ]

    for size_codebooks, codebook_specs in zip(all_size_of_codebooks, all_codebook_specs):
        for delta in delta_values:
            print("="*80)
            print(f"[INFO] Starting simulation for codebooks {codebook_specs} with Δ={delta}")
            print("="*80)

            parameters = Parameters(    N_R=64, 
                                        N_T=1, 
                                        N_RIS=100, 
                                        size_codebooks=size_codebooks, 
                                        codebook_specs=codebook_specs,
                                        mean_noise=0,
                                        SNR=10,
                                        # snr_values = [0,5,10,20],
                                        snr_values = [20],
                                        modification_channel=0,
                                        type_channel="half-spaced ULAs",
                                        type_modulation="BPSK", 
                                        mean_channel=0, 
                                        std_channel=[], 
                                        sigma_alpha=0, 
                                        gamma=0.94,
                                        learning_rate_init=5e-5,
                                        params_list=[256,256],
                                        batch_size=1024,
                                        replay_buffer_memory_size=120000,
                                        max_norm=0.5,
                                        do_gradient_clipping = True,
                                        loss_fct='mse',
                                        n_epochs=10000,
                                        # n_epochs = 5,
                                        n_time_steps_dqn=64,
                                        n_channels_train_DQN=5,
                                        # n_channels_train_DQN=1,
                                        freq_update_target=20,
                                        tau = 0.05,
                                        max_len_path=20,
                                        epsilon=1,
                                        epsilon_decay=0.992,
                                        epsilon_min=0.01,
                                        delta_init=delta,
                                        delta_decay=1,
                                        train_or_test=True,
                                        Train_Deep_Q_Learning=True,
                                        saving_freq_DQN=500,
                                        # saving_freq_DQN=1,
                                        test_freq_DQN=1,
                                        Train_Q_Learning=True,
                                        n_episodes=10000,
                                        # n_episodes=5,
                                        n_time_steps_ql=64,
                                        n_channels_train_QL=5,
                                        len_path=20,
                                        saving_freq_QL = 500,
                                        # saving_freq_QL = 1,
                                        test_freq_QL = 1,
                                        delta_final=5e-2, 
                                        len_window_action=1, 
                                        len_window_channel=10, 
                                        precision=2,  
                                        blabla_other_states=1, 
                                        min_representatives_q_learning_train=100, 
                                        min_representatives_q_learning_test=10,
                                        learning_rate_decay = 0.99,
                                        learning_rate_min = 1e-4
                                        
                                        ) # All the parameters stored in a class

            # Physical system Setup

            # The channel model
            channel = Channel(
                parameters
            ) 
            
            # The transmitted signal
            signal = Signal(
                parameters
            ) 
            
            # The codebook for the RIS
            codebooks = Codebooks(
                parameters
            ) 

            # The feedback function between the receiver and the transmitter
            feedback = Feedback(
                parameters,
                channel,
                codebooks,
                signal
                ) 

            paths = ExperimentPaths.make_new_experiment_folder(base_dir = "./Data")
            store = Store(
                paths
                )

            dataset_factory = DatasetFactory()
            dataset_proba, parameters, codebooks = dataset_factory.get_dataset(
                dataset_mode=DatasetMode.GENERATE, 
                store=store, 
                parameters=parameters, 
                channel=channel, 
                codebooks=codebooks, 
                feedback=feedback,
                noisy_samples=True
                )

            noise_factory = NoiseFactory()
            noise_parameters = noise_factory.get_noise(
                noise_mode=NoiseMode.ANALYTICAL, 
                paths=paths.root,
                feedback=feedback,
                channel=channel, 
                parameters=parameters, 
                )

            probability = Probability(
                parameters, 
                dataset_proba, 
                noise_parameters
                )

            # States space 
            states = State(
                parameters
                )

            environment = Environment(
                states=states,
                parameters=parameters,
                probability=probability,
                dataset_train=dataset_proba,
                dataset_test=dataset_proba
                )

            runner = Runner(
                parameters= parameters,
                environment= environment, 
                store= store, 
                probability= probability
                )
            
            # Train Q-Learning and get the policy
            q_policy = runner.run_q_learning()

            # Train Deep Q-Learning and get the policy
            dqn_agent, dqn_policy = runner.run_deep_q_learning()
            
            runner.run_testing(q_policy, dqn_agent, dqn_policy)

            print(f"[INFO] Simulation for Example_{paths.root.split('_')[-1]} Completed.")

if __name__ == "__main__":
    main()
