import torch
import random
import numpy as np 
from tqdm import tqdm
from config.parameters import Parameters
from reinforcement_learning.env import Environment 
from experiments.store import ExperimentPaths

from reinforcement_learning.deep_q_learning.components.network import DQN
from reinforcement_learning.deep_q_learning.components.replay_buffer import ReplayBuffer
from reinforcement_learning.deep_q_learning.components.schedules import multiplicativeDecaySchedule

from reinforcement_learning.deep_q_learning.algo.dqn_update import dqn_learn_update_step, dqn_target_update
from reinforcement_learning.deep_q_learning.utils import (
    save_dqn_weights, 
    plot_Convergence, 
    save_Data,
    save_dqn_training_state,
    load_dqn_training_state
)

class DeepQLearningAgent():
    """ Class for Deep Q-Learning Algorithm"""
    
    def __init__(
        self,
        parameters: Parameters,
        environment : Environment,
        paths: ExperimentPaths
    ):
        
        self.simu_parameters = parameters
        self.environment = environment
        self.paths = paths
        
        self.input_dims = self.environment.get_size_states()
        self.n_states = self.environment.state_space.get_n_states()
        self.n_actions = self.environment.get_size_codebook()[1]
        
        ## Parameters
        params_dict = self.simu_parameters.get_dqn_parameters()
        
        self.saving_freq = params_dict["saving_freq"]
        
        self.n_epochs = params_dict["n_epochs"]
        self.n_channels_train = params_dict["n_channels_train"]
        self.n_time_steps = params_dict["n_time_steps"]
        self.max_len_path = params_dict["max_len_path"]
        
        self.batch_size = params_dict["batch_size"]
        replay_buffer_memory_size = params_dict["replay_buffer_memory_size"]

        self.gamma = params_dict["gamma"]
        self.max_norm = params_dict["max_norm"]
        self.do_gradient_clipping = params_dict["do_gradient_clipping"]

        self.freq_update_target = params_dict["freq_update_target"]
        self.tau = params_dict["tau"]
        self.targetNet_update_method = params_dict["targetNet_update_method"]

        learning_rate = params_dict["learning_rate_init"]
        epsilon_init = params_dict["epsilon_init"]
        epsilon_decay = params_dict["epsilon_decay"]
        epsilon_min = params_dict["epsilon_min"]
        delta_init = params_dict["delta_init"]
        delta_decay = params_dict["delta_decay"]
        delta_min = params_dict["delta_min"] # Final degree of precision we want to reach
        
        self.saving_freq = params_dict["saving_freq"]

        self.continue_training = params_dict["continue_training"]

        self.start_epoch = 0
        self.update_step = 0 # Count how many gradients have been performed

        # ---- Dataset ----
        self.dataset_train = self.environment.get_dataset()

        # ---- Device ----
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ---- Networks ----
        self.evaluation_q_network = DQN(
                                        self.simu_parameters,
                                        self.input_dims,
                                        self.n_actions)
        
        self.target_q_network = DQN(
                                    self.simu_parameters,
                                    self.input_dims,
                                    self.n_actions)
        
        # Move networks to device
        self.evaluation_q_network.to(self.device)
        self.target_q_network.to(self.device)

        # the target network is initialized with the same weights as the evaluation network
        self.target_q_network.load_state_dict(self.evaluation_q_network.state_dict())
        
        # ---- Optimizer ----
        # The network is less pure with the optimizer, it needs to be in the agent.
        self.optimizer = torch.optim.Adam(
                                            self.evaluation_q_network.parameters(), # Pass parameters not module object
                                            learning_rate,
                                            weight_decay=1e-4)

        # ---- Replay Buffer ----
        self.replay_buffer = ReplayBuffer(
                                            replay_buffer_memory_size,
                                            self.batch_size, 
                                            self.input_dims)
        
        # ---- Schedules ----
        self.epsilon_schedule = multiplicativeDecaySchedule(
            init_value = epsilon_init,
            decay = epsilon_decay,
            min_value = epsilon_min
        )
        
        self.delta_schedule = multiplicativeDecaySchedule(
                                                            init_value = delta_init,
                                                            decay = delta_decay,
                                                            min_value = delta_min
                                                            )

        # ---- Policy  ----
        self.policy = np.zeros([self.n_states, self.n_actions])
        
    def choose_action(self, state, epsilon: float):
        """
        Choose an action following Epsilon Greedy Policy.
        =======
        Args:
        =======
        @ state : The array/tensor of the current state
        @ epsilon : Epsilon schedule

        Returns:
        The index of the action to choose next
        """
        if np.random.random() < epsilon :
            action = np.random.randint(self.n_actions)
        else :
            state_tensor = torch.as_tensor(state, dtype= torch.float32, device= self.device)
            with torch.no_grad():
                q_values = self.evaluation_q_network(state_tensor)
                action = torch.argmax(q_values).item()
        return action

    def load_training_state(self):
        checkpoint = load_dqn_training_state(self.paths, self.device)

        self.evaluation_q_network.load_state_dict(checkpoint["eval_state_dict"])
        self.target_q_network.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.update_step = checkpoint["update_step"]

        # Restore schedules
        self.epsilon_schedule.value = checkpoint["epsilon_schedule"]["value"]
        self.delta_schedule.value = checkpoint["delta_schedule"]["value"]

        # Restore replay buffer
        rb = checkpoint["replay_buffer"]
        self.replay_buffer.current_state_memory = rb["current_state_memory"]
        self.replay_buffer.next_state_memory = rb["next_state_memory"]
        self.replay_buffer.action_memory = rb["action_memory"]
        self.replay_buffer.reward_memory = rb["reward_memory"]
        self.replay_buffer.terminated_memory = rb["terminated_memory"]
        self.replay_buffer.memory_counter = rb["memory_counter"]
        self.replay_buffer.idx = rb["idx"]

        rng_state = checkpoint.get("rng_state", None)
        if rng_state is not None:
            random.setstate(rng_state["python_random"])
            np.random.set_state(rng_state["numpy_random"])
            torch.set_rng_state(rng_state["torch_cpu"])

            if torch.cuda.is_available() and rng_state["torch_cuda"] is not None:
                torch.cuda.set_rng_state_all(rng_state["torch_cuda"])

        self.start_epoch = checkpoint["epoch"] + 1

        history = checkpoint["history"]
        return {
            "avg_losses": history["avg_losses"],
            "avg_len_path": history["avg_len_path"],
            "avg_rewards": history["avg_rewards"],
            "epsilons": history["epsilons"],
            "avg_grad_norms": history["avg_grad_norms"],
            "max_grad_norms": history["max_grad_norms"],
        }
    
    def train_one_epoch(self, epsilon: float):
        """ 
        Run one training epoch
        =======
        Args:
        =======
        @ epsilon : for the Epsilon Greedy Policy
        @ delta: for the stopping criteria

        Returns:
        @ avg_loss : the average of the loss during the training
        @ avg_len_path : the average length of path after all the channel realization
        @ n_updates : the amount of times the Target Network has been updated 
        """
        
        epoch_losses = []
        epoch_path_lengths = []
        epoch_rewards = []
        epoch_grad_norms = []

        #---------------------------------------- Loop over channel realizations --------------------------------------------------

        for _ in range(self.n_channels_train):
                                
                # Training : Pick a random channel from dataset
                index_class_channel = np.random.randint(0,len(self.dataset_train)) # Random class
                index_specific_channel = np.random.randint(0,len((self.dataset_train)[index_class_channel][1])) # Random channel from this class

                index_channel = (index_class_channel,index_specific_channel)

                #---------------------------------------- Loop over time steps --------------------------------------------------

                for _ in range(self.n_time_steps):
                    
                    """ -----------------------------Modification during last test-------------------------------------------- """
                    ## Always start from the initial state
                    current_state = self.environment.state_space.get_state_from_index(0) 
                    
                    ## Set the prior at the initial state
                    self.environment.reset_prior()
                    """ ------------------------------------------------------------------------- """
                    ## Track path length as number of actions are taken to measure how efficient the policy is
                    path_len = 0
                    episode_reward = 0.0
                
                    #---------------------------------------- Loop until reaching the maximum length of path allowed --------------------------------------------------

                    for path in range(self.max_len_path):

                        # Action selection
                        index_action = self.choose_action(
                            current_state, 
                            epsilon
                        )
                        
                        # Environment step    
                        next_state, reward, terminated, truncated, _ = self.environment.step(
                            index_channel,
                            index_action,
                            model_type = 'DQN'
                            )
                        
                        # Update the path length and reward trackers for this episode
                        path_len += 1
                        episode_reward += reward

                        # Setting the Gymnasium-style step.
                        time_limit_reached = (path == self.max_len_path - 1)
                        truncated = (time_limit_reached and not terminated)

                        episode_over  = terminated or truncated
                            
                        # Store the transition in the replay buffer training
                        self.replay_buffer.store_transition(
                            current_state, 
                            index_action, 
                            reward, 
                            next_state, 
                            terminated
                        )
                            
                        current_state = next_state
                        if episode_over:
                            break

                    epoch_path_lengths.append(path_len)
                    epoch_rewards.append(episode_reward)
                    
                    #------------------- Learning step (only if the buffer has enough samples) ------------------
                    if self.replay_buffer.memory_counter >= self.batch_size:
                        # Sample a batch from the Replay Buffer
                        current_state_batch, action_batch, reward_batch, next_state_batch, terminated_batch = self.replay_buffer.sample_buffer()
                        
                        # Start the training process
                        loss_value, grad_norm_before_clip  = dqn_learn_update_step(
                            self.evaluation_q_network,
                            self.target_q_network,
                            self.optimizer, 
                            current_state_batch, 
                            action_batch, 
                            reward_batch, 
                            next_state_batch,
                            terminated_batch,
                            self.gamma, 
                            self.do_gradient_clipping, 
                            self.max_norm
                        )
                        
                        epoch_losses.append(loss_value)
                        epoch_grad_norms.append(grad_norm_before_clip)

                        self.update_step += 1

                        #------------------- Target network update (frequency in gradient steps) ------------------
                        if self.targetNet_update_method.lower() == "soft": 
                            # tau is typically small : 1e-3 to 5e-3 (sometimes 1e-2). Soft update is already “gentle”. Doing it every step is standard and stable.
                            dqn_target_update(
                                self.evaluation_q_network, 
                                self.target_q_network, 
                                self.tau, 
                                self.targetNet_update_method
                            )

                        if self.targetNet_update_method.lower() == "hard": 
                            if self.update_step % self.freq_update_target == 0:
                                dqn_target_update(
                                    self.evaluation_q_network, 
                                    self.target_q_network, 
                                    self.tau, 
                                    self.targetNet_update_method
                            )
        return {
            "avg_loss": np.mean(epoch_losses) if len(epoch_losses) > 0 else np.nan,
            "avg_len_path" : np.mean(epoch_path_lengths),
            "avg_reward": np.mean(epoch_rewards),
            "avg_grad_norm_before_clip": np.mean(epoch_grad_norms) if len(epoch_grad_norms) > 0 else np.nan,
            "max_grad_norm_before_clip": np.max(epoch_grad_norms) if len(epoch_grad_norms) > 0 else np.nan,
            "n_updates": self.update_step
        }
                
    
    def train(self):
        """" 
        Train the Deep Q-Learning Network, with optional recovery.
        """
        
        epsilons = []
        avg_losses = []
        avg_len_path = []
        avg_rewards = []
        avg_grad_norms = []
        max_grad_norms = []

        # Reset the values of epsilon and delta
        if self.continue_training:
            print("[INFO] Recovering DQN training from saved state...")
            history = self.load_training_state()
            
            epsilons = history["epsilons"]
            avg_losses = history["avg_losses"]
            avg_len_path = history["avg_len_path"]
            avg_rewards = history["avg_rewards"]
            avg_grad_norms = history["avg_grad_norms"]
            max_grad_norms = history["max_grad_norms"]

            epsilon = self.epsilon_schedule.get()
            delta = self.delta_schedule.get()
            self.environment.set_delta_current(delta)

            print(f"[INFO] Recovery start epoch = {self.start_epoch}")
            print(f"[INFO] Remaining epochs = {self.n_epochs - self.start_epoch}")
        else:
            self.start_epoch = 0
            self.epsilon_schedule.reset()
            self.delta_schedule.reset()

            epsilon = self.epsilon_schedule.get()
            delta = self.delta_schedule.get()
            self.environment.set_delta_current(delta)

            print(f"Initialization of epsilon and delta schedules : epsilon_init = {epsilon} delta_init = {delta}")

        print(f"Initialize Training for DQN Network on Device : {self.device}")
        
        #---------------------------------------- Loop over epochs --------------------------------------------------

        for epoch in tqdm(range(self.start_epoch, self.n_epochs)):
            one_epoch_metrics = self.train_one_epoch(
                epsilon= epsilon
            )
            
            epsilons.append(epsilon)
            avg_losses.append(one_epoch_metrics["avg_loss"])
            avg_len_path.append(one_epoch_metrics["avg_len_path"])
            avg_rewards.append(one_epoch_metrics["avg_reward"])
            avg_grad_norms.append(one_epoch_metrics["avg_grad_norm_before_clip"])
            max_grad_norms.append(one_epoch_metrics["max_grad_norm_before_clip"])

            tqdm.write(
                        f"Epoch {epoch}:" 
                        f"Loss = {one_epoch_metrics['avg_loss']:.6f}, "
                        f"Avg Path = {one_epoch_metrics['avg_len_path']:.2f}, "
                        f"Avg Reward = {one_epoch_metrics['avg_reward']:.2f}, "
                        f"Avg Grad Norm = {one_epoch_metrics['avg_grad_norm_before_clip']:.4f}, "
                        f"Max Grad Norm = {one_epoch_metrics['max_grad_norm_before_clip']:.4f}, "
                        f"Epsilon = {epsilon:.3f}, "
                        f"Delta = {delta: .4e}"
                    )

            print(f'[INFO] Value of epsilon in the agent before the update:  epsilon = {epsilon}')
            print(f'[INFO] Value of delta in the agent before the update:  delta_agent = {delta}')
            print(f'[INFO] Value of delta in the environment before the update:  delta_env = {self.environment.get_delta_current()}')

            # Update Epsilon and Delta values in the agent
            epsilon = self.epsilon_schedule.step()
            delta = self.delta_schedule.step()
            
            # Update Delta value in the environment
            self.environment.set_delta_current(delta)

            print(f'[INFO] Epsilon value updated : {epsilon}')
            print(f'[INFO] Delta value updated in agent : delta_agent = {delta}')
            print(f'[INFO] Delta value updated in environment : delta_env = {self.environment.get_delta_current()}')
        
            # save the model checkpoints
            if epoch % self.saving_freq == 0:
                save_dqn_weights(
                    self.paths, 
                    epoch, 
                    self.evaluation_q_network, 
                    self.target_q_network
                )

                save_dqn_training_state(
                    self.paths,
                    epoch,
                    self.evaluation_q_network,
                    self.target_q_network,
                    self.optimizer,
                    self.replay_buffer,
                    self.epsilon_schedule,
                    self.delta_schedule,
                    self.update_step,
                    avg_losses,
                    avg_len_path,
                    avg_rewards,
                    epsilons,
                    avg_grad_norms,
                    max_grad_norms,
                )

        # save the model checkpoints for the last epoch
        save_dqn_weights(
            self.paths, 
            "last", 
            self.evaluation_q_network, 
            self.target_q_network,
            final_epoch = self.n_epochs
        )

        save_dqn_training_state(
            self.paths,
            self.n_epochs - 1,
            self.evaluation_q_network,
            self.target_q_network,
            self.optimizer,
            self.replay_buffer,
            self.epsilon_schedule,
            self.delta_schedule,
            self.update_step,
            avg_losses,
            avg_len_path,
            avg_rewards,
            epsilons,
            avg_grad_norms,
            max_grad_norms,
        )

        # Save the records of average losses, length of path and epsilon
        save_Data(
            self.paths,
            avg_losses,
            avg_len_path,
            avg_rewards,
            epsilons,
            avg_grad_norms,
            max_grad_norms
        )
        
        # Plot the convergence of the algorithm
        plot_Convergence(
            self.paths,
            avg_losses,
            avg_len_path,
            avg_rewards,
            avg_grad_norms,
            max_grad_norms
        )
        
        # The NN trained is a function that must find the optimal policy
        return self.evaluation_q_network
        
