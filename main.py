from config.parameters import Parameters
from systemModel.codebooks import CodebookSpec

from experiments.store import Store, ExperimentPaths
from experiments.builder import ExperimentBuilder

from reinforcement_learning.deep_q_learning.components.seed import set_seed

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
                    1e-1, # session DQL Example_6
                    # 1e-2, # session dql Example_5
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

                                        SNR=10,
                                        # snr_values = [0,5,10,20],
                                        snr_values = [20],
                                        type_channel="half-spaced ULAs",
                                        type_modulation="BPSK", 
                                        mean_noise=0,
                                        mean_channel=0, 
                                        std_channel=[], 
                                        sigma_alpha=0, 

                                        gamma=0.94,
                                        _greedy_mode = False,

                                        learning_rate_init=1e-4,
                                        learning_rate_decay = 0.99,
                                        learning_rate_min = 1e-6,

                                        epsilon_init=1,
                                        epsilon_decay=0.992,
                                        epsilon_min=0.01,

                                        delta_init=delta,
                                        # delta_decay=0.99,
                                        delta_decay=1,
                                        delta_min=1e-3, 

                                        params_list=[256,256],
                                        loss_fct='mse',
                                        batch_size=256,
                                        replay_buffer_memory_size=120000,

                                        max_norm=0.5,
                                        do_gradient_clipping = True,

                                        # n_epochs=1000,
                                        n_epochs = 2,
                                        n_time_steps_dqn=64,
                                        # n_channels_train_DQN=5,
                                        n_channels_train_DQN=1,
                                        
                                        # n_episodes=1000,
                                        n_episodes=2,
                                        n_time_steps_ql=64,
                                        # n_channels_train_QL=5,
                                        n_channels_train_QL=1,
                                        max_len_path=20,
                                        len_path=20,
                                        
                                        tau = 0.005,
                                        freq_update_target=100,
                                        targetNet_update_method = "soft",
                                        
                                        Train_Deep_Q_Learning=True,
                                        # saving_freq_DQN=50,
                                        saving_freq_DQN=1,
                                        test_freq_DQN=1,

                                        Train_Q_Learning=True,
                                        # saving_freq_QL = 50,
                                        saving_freq_QL = 1,
                                        test_freq_QL = 1,
                                        
                                        precision=2,  
                                        len_window_channel=10,
                                        modification_channel=0,
                                        min_representatives_q_learning_train=100,
                                        min_representatives_q_learning_test=10
                                        ) # All the parameters stored in a class

            paths = ExperimentPaths.make_new_experiment_folder(base_dir = "./Data")

            store = Store(paths)

            print(f"\n[EXPERIMENT] root = {store.paths.root}")
            print(f"[EXPERIMENT] dataset = {store.paths.dataset_pickle}")
            print(f"[EXPERIMENT] noise   = {store.paths.noise_csv}\n")
            
            # ----------------------------------- Build full pipeline --------------------------------

            builder = ExperimentBuilder(
                parameters= parameters,
                store= store
            )

            runner = builder.build()

            # -------------------------------------- Train / Test --------------------------------------

            # Train Q-Learning and get the policy
            q_policy = runner.run_q_learning()

            # Train Deep Q-Learning and get the policy
            dqn_out = runner.run_deep_q_learning()
            if dqn_out is None:
                dqn_agent, dqn_policy = None, None
            else:
                dqn_agent, dqn_policy = dqn_out 
            
            runner.run_testing(q_policy, dqn_agent, dqn_policy)

            print(f"[INFO] Simulation for Example_{paths.root.split('_')[-1]} Completed.")

if __name__ == "__main__":
    main()
