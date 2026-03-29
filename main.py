from config.parameters import Parameters
from systemModel.codebooks import CodebookSpec

from experiments.store import Store, ExperimentPaths
from experiments.builder import ExperimentBuilder
from experiments.runner import Runner

from experiments.async_eval import start_async_eval_worker

from reinforcement_learning.deep_q_learning.components.seed import set_seed

def main():
    all_size_of_codebooks = [
                                [8, 14], # session dqn lame20 from 40 et lame21 from 44
                                # [16, 30], # session DQN lame22 from 41 et lame25 from 45
                                # [32, 62],
                                # [64, 126]
                            ]

    all_codebook_specs = [
                            # [
                            #     CodebookSpec(kind="narrow", N=8),
                            #     CodebookSpec(kind="hierarchical", K=3, M=2)
                            # ],
                            # [
                            #     CodebookSpec(kind="narrow", N=16),
                            #     CodebookSpec(kind="hierarchical", K=4, M=2)
                            # ],
                            [
                                CodebookSpec(kind="narrow", N=8),
                                CodebookSpec(kind="dft", N=14)
                            ],
                            # [
                            #     CodebookSpec(kind="narrow", N=16),
                            #     CodebookSpec(kind="dft", N=30)
                            # ]
                        ]

    delta_values = [
                    # 1e-1, # session DQL Example_6 , 29
                    # 2e-1,
                    # 3e-1,
                    1e-2, # session dql Example_5 , 30, 39, 40
                    # 2e-2,
                    ]

    base_seed = 42

    for i, (size_codebooks, codebook_specs) in enumerate(zip(all_size_of_codebooks, all_codebook_specs)):
        for j, delta in enumerate(delta_values):
            comm_kind = codebook_specs[0].kind
            pilot_kind = codebook_specs[1].kind
            set_seed(base_seed)
            print("="*80)
            print(f"[INFO] Starting simulation for codebooks {codebook_specs} with delta = {delta}")
            print("="*80)

            parameters = Parameters(    
                experiment_note = (
                    # "Hierarchical baseline tested with noisy pilot measurements."
                    f"Communication codebook = {comm_kind} ({size_codebooks[0]} codewords). "
                    f"Pilot codebook = {pilot_kind} ({size_codebooks[1]} codewords)."
                    f"Delta = {delta}. Tau = 0.001."
                ),
                experiment_seed = base_seed,

                N_R = 64, 
                N_T = 1, 
                N_RIS = 100, 

                size_codebooks = size_codebooks, 
                codebook_specs = codebook_specs,

                SNR = 10,
                # snr_values = [0,5,10,20],
                snr_values = [20],
                type_channel = "half-spaced ULAs",
                type_modulation = "BPSK", 
                mean_noise = 0,
                mean_channel = 0, 
                std_channel=  [], 
                sigma_alpha = 0, 
                hierarchical_noisy_measurement = False,

                gamma = 0.94,
                _greedy_mode = False,

                learning_rate_init = 1e-4,
                learning_rate_decay = 0.99,
                learning_rate_min = 1e-6,

                epsilon_init = 1,
                epsilon_decay = 0.992,
                epsilon_min = 0.01,

                delta_init = delta,
                # delta_decay = 0.99,
                delta_decay = 1,
                delta_min = 1e-3, 

                params_list = [256,256],
                loss_fct = 'mse',
                batch_size = 256,
                replay_buffer_memory_size = 120000,

                # n_epochs=10000,
                n_epochs = 10,
                n_time_steps_dqn = 64,
                n_channels_train_DQN = 5,
                # n_channels_train_DQN=1,
                
                # n_episodes = 10000,
                n_episodes = 10,
                n_time_steps_ql = 64,
                n_channels_train_QL = 5,
                # n_channels_train_QL = 1,
                max_len_path = 20,
                len_path = 20,

                max_norm = 0.5,
                do_gradient_clipping = True,
                
                tau = 0.001,
                freq_update_target = 100,
                targetNet_update_method = "soft",
                
                Train_Deep_Q_Learning = True,
                Train_Q_Learning = True,
                # saving_freq = 500,
                saving_freq = 2,
                test_freq = 1,
                # test_freq = 500,

                continue_training = False,
                recover_checkpoint_path = "./Data/Example_34",

                enable_async_eval = True,
                async_eval_poll_seconds = 10,
                async_eval_device = "cpu",
                
                precision = 2,  
                # len_window_channel = 10,
                len_window_channel = 14,
                # len_window_channel = 30,
                modification_channel = 0,
                min_representatives_q_learning_train = 100,
                min_representatives_q_learning_test = 10
            ) 

            print(f"\nExperiment note: {parameters.experiment_note}\n")

            if parameters.continue_training:
                if parameters.recover_checkpoint_path is None:
                    raise ValueError("continue_training = True but recover_checkpoint_path is None")
                paths = ExperimentPaths(root=parameters.recover_checkpoint_path)
            else:
                paths = ExperimentPaths.make_new_experiment_folder(base_dir="./Data")

            store = Store(paths)

            # Save experiment parameters
            if parameters.continue_training:
                params_path = store.paths.params_resume_training_file("both")
            else:
                params_path = store.paths.params_file("both")

            parameters.save_to_file(params_path, params_type="both")

            print(f"\n[EXPERIMENT] root = {store.paths.root}")
            print(f"[EXPERIMENT] params  = {params_path}")
            print(f"[EXPERIMENT] dataset = {store.paths.dataset_pickle}")
            print(f"[EXPERIMENT] noise   = {store.paths.noise_csv}\n")
            
            # ----------------------------------- Build full pipeline --------------------------------

            builder = ExperimentBuilder(
                parameters= parameters,
                store= store
            )

            context = builder.build()
            runner = Runner(
                context = context
            )

            # -------------------------------------- Train / Test --------------------------------------

            # Train Q-Learning and get the policy
            q_policy = runner.run_q_learning()

            # Async evaluation process handle (background tester) and its stop signal
            async_process = None
            stop_event = None
            dqn_agent, dqn_policy = None, None
            training_error = None

            try:
                # Start background evaluation only if async eval is enabled
                if parameters.enable_async_eval and parameters.Train_Deep_Q_Learning:
                    mode = "both" if q_policy is not None else "dqn"

                    # Launch the background evaluation worker
                    # It watches saved checkpoints and tests them while training continues
                    stop_event, async_process = start_async_eval_worker(
                        parameters = parameters,
                        experiment_root = store.paths.root,
                        mode = mode,
                        q_policy = q_policy
                    )

                # Train Deep Q-Learning and get the policy
                dqn_out = runner.run_deep_q_learning()
                if dqn_out is not None:
                    dqn_agent, dqn_policy = dqn_out
            
            except Exception as e:
                training_error = e
                print(f"[ERROR] Training failed before final testing: {e}")

            finally:
                # Make sure the background evaluator is stopped cleanly,
                # even if training crashes or is interrupted
                if async_process is not None:
                    print("[MAIN] stopping async worker...")
                    stop_event.set()

                    print("[MAIN] waiting up to 10s for async worker...")
                    async_process.join(timeout=10)

                    if async_process.is_alive():
                        print("[MAIN] async worker still busy -> terminating it.")
                        async_process.terminate()
                        async_process.join()

                    print("[MAIN] async worker joined.")
                        
            try:
                runner.run_testing(q_policy, dqn_agent, dqn_policy)
            except Exception as e:
                print(f"[ERROR] Final testing failed: {e}")

            if training_error is not None:
                raise training_error
            
            print(f"[INFO] Simulation for Example_{paths.root.split('_')[-1]} Completed.")

if __name__ == "__main__":
    main()
