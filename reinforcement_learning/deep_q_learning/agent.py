import numpy as np 
from tqdm import tqdm

import os
import torch
import torch.nn as nn

from config.parameters import Parameters
from reinforcement_learning.env import Environment 
from reinforcement_learning.deep_q_learning.components.network import DQN
from reinforcement_learning.deep_q_learning.components.replay_buffer import ReplayBuffer
from reinforcement_learning.deep_q_learning.components.schedules import multiplicativeDecaySchedule

from reinforcement_learning.deep_q_learning.algo.dqn_update import dqn_learn_update_step, dqn_target_update
from reinforcement_learning.deep_q_learning.utils import save_model_checkpoints, save_last_models, plot_Convergence, save_Data


class DeepQLearningAgent():
    """ Class for Deep Q-Learning Algorithm"""
    
    def __init__(self,
                    input_dims: int, #we input the state index
                    environment : Environment,
                    parameters: Parameters,
                    name_file:str="Data/Example_0/DQN"
                ):
        
        self.simu_parameters = parameters
        self.name = name_file
        self.environment = environment
        self.input_dims = self.environment.get_size_states()
        self.n_states = self.environment.state_space.get_n_states()
        self.n_actions = self.environment.get_size_codebook()[1]
        
        ## Parameters
        params_dict = self.simu_parameters.get_dqn_parameters()
        learning_rate = params_dict["learning_rate_init"]
        batch_size = params_dict["batch_size"]
        replay_buffer_memory_size = params_dict["replay_buffer_memory_size"]
        epsilon_init = params_dict["epsilon_init"]
        epsilon_decay = params_dict["epsilon_decay"]
        epsilon_min = params_dict["epsilon_min"]
        delta_init = params_dict["delta_init"]
        delta_decay = params_dict["delta_decay"]
        delta_min = params_dict["delta_min"] # Final degree of precision we want to reach

        # ---- Dataset ----
        self.dataset_train,self.dataset_test = environment.get_dataset()

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
                                            batch_size, 
                                            input_dims)
        
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
        """ Choose an action following the Epsilon Greedy Policy"""

        if np.random.random() < epsilon :
            action = np.random.randint(self.n_actions)
        else :
            state_tensor = torch.as_tensor(state, dtype= torch.float32, device= self.device)
            with torch.no_grad():
                q_values = self.evaluation_q_network(state_tensor)
                action = torch.argmax(q_values).item()
        return action
    
    def train_one_epoch(self, epsilon: float, delta: float, params_dict={}):
        """ 
        Run one training epoch
        =======
        Args:
        =======
        @ epsilon : for the Epsilon Greedy Policy
        @ delta: for the stopping criteria
        @ params_dict: access to dictionary for needed parameters

        Returns:
        @ avg_loss : the average of the loss during the training
        @ avg_len_path : the average length of path after all the channel realization
        @ n_updates : the amount of times the Target Network has been updated """

        batch_size = params_dict["batch_size"]
        
        n_channels_train = params_dict["n_channels_train"]
        n_time_steps = params_dict["n_time_steps"]
        max_len_path = params_dict["max_len_path"]
        train_or_test = params_dict["train_or_test"]
        
        freq_update_target = params_dict["freq_update_target"]
        tau = params_dict["tau"]
        targetNet_update_method = params_dict["targetNet_update_method"]

        gamma = params_dict["gamma"]
        max_norm = params_dict["max_norm"]
        do_gradient_clipping = params_dict["do_gradient_clipping"]
        
        epoch_losses = []
        epoch_path_lengths = []

        # Count how many gradients have been performed
        update_step = 0

        # Ensure the environment uses the right delta
        self.environment.set_delta_current(delta) 

        print(f"Value of delta in the environment :  delta = {self.environment.get_delta_current}")

        #---------------------------------------- Loop over channel realizations --------------------------------------------------

        for channel_realization in range(n_channels_train):
                
                ## Always start from the initial state
                current_state = self.environment.state_space.get_state_from_index(0) 
                
                ## Set the prior at the initial state
                self.environment.reset_prior()
                len_window_channel = self.environment.get_len_window_channel()
                
                # Pick a new channel from dataset 
                if train_or_test:
                    # Training : Random channel
                    index_class_channel = np.random.randint(0,len(self.dataset_train)) # Random class
                    index_specific_channel = np.random.randint(0,len((self.dataset_train)[index_class_channel][1])) # Random channel from this class
                else:
                    ## Testing : test elements in the dataset one after the other (???????????? I don't understand this part)
                    index_class_channel = channel_realization%len(self.dataset_test)
                    index_specific_channel = channel_realization//len(self.dataset_test)
                    if channel_realization//len(self.dataset_test) >= len((self.dataset_test)[index_class_channel][1]):
                        index_specific_channel = np.random.randint(0,len((self.dataset_test)[index_class_channel][1]))

                index_channel = (index_class_channel,index_specific_channel)

                #---------------------------------------- Loop over time steps --------------------------------------------------

                for _ in range(n_time_steps):
                    
                    ## Track path length as number of actions are taken to measure how efficient the policy is
                    path_len = 0 
                
                    #---------------------------------------- Loop until reaching the maximum length of path allowed --------------------------------------------------

                    for path in range(max_len_path):

                        # Reset the curse of dimension : reset the list of samples and put prior = posterior
                        if path % len_window_channel == 0:
                            self.environment.reset_curse_dimension()

                        # Action selection
                        if train_or_test:
                            index_action = self.choose_action(current_state, epsilon)
                        else:
                            index_action = self.choose_action(current_state, 0.0) ## Greedy Action during the test
                        
                        # Environment step    
                        next_state, reward = self.environment.step(
                                                                    index_channel,
                                                                    index_action,
                                                                    train_or_test=train_or_test,
                                                                    model_type = 'DQN'
                                                                    )
                        
                        # Save the reward
                        path_len += 1
                            
                        # Store the transition in the replay buffer training
                        if train_or_test:
                            self.replay_buffer.store_transition(current_state, index_action, reward, next_state)
                            
                        current_state = next_state
                        
                        delta_current = self.environment.get_delta_current()
                        if train_or_test:
                            if max(current_state) >= 1 - delta_current:
                                break 
                        else:
                            if max(self.environment.get_posterior()) >= 1 - delta_current:
                                break
                        
                        print(f'Delta value in env = {self.environment.get_delta_current()}')
                        print(f'Delta value in agent = {delta}')

                    epoch_path_lengths.append(path_len)
                    
                    #------------------- Learning step (only if the buffer has enough samples) ------------------
                    if self.replay_buffer.memory_counter >= batch_size:
                        # Sample a batch from the Replay Buffer
                        current_state_batch, action_batch, reward_batch, next_state_batch = self.replay_buffer.sample_buffer()
                        
                        # Start the training process
                        loss_value = dqn_learn_update_step(
                                                            self.evaluation_q_network,
                                                            self.target_q_network,
                                                            self.optimizer, 
                                                            current_state_batch, 
                                                            action_batch, 
                                                            reward_batch, 
                                                            next_state_batch, 
                                                            gamma, 
                                                            do_gradient_clipping, 
                                                            max_norm )
                        epoch_losses.append(loss_value)

                        update_step += 1

                        #------------------- Target network update (frequency in gradient steps) ------------------
                        if update_step % freq_update_target == 0:
                            dqn_target_update(
                                                self.evaluation_q_network, 
                                                self.target_q_network, 
                                                tau, 
                                                targetNet_update_method
                                                )
        return {
            "avg_loss": np.mean(epoch_losses) if len(epoch_losses) > 0 else np.nan,
            "avg_len_path" : np.mean(epoch_path_lengths),
            "n_updates": update_step
        }
                
    
    def train(self, params_dict = {}, testing_objects_dict = {}):
        """" Train the Deep Q-Learning Network """
        
        n_epochs = params_dict["n_epochs"]
        saving_freq = params_dict["saving_freq"]

        # Reset the values of epsilon and delta
        self.epsilon_schedule.reset()
        self.delta_schedule.reset()

        epsilon = self.epsilon_schedule.get()
        delta = self.delta_schedule.get()
        print(f"Initialization of epsilon and delta schedules : epsilon_init = {epsilon} delta_init = {delta}")

        print(f"Initialize Training for DQN Network on Device : {self.device}")
        
        avg_losses = []
        avg_len_path = []
        epsilons = []

        #---------------------------------------- Loop over epochs --------------------------------------------------

        for epoch in tqdm(range(n_epochs)):
            
            one_epoch_metrics = self.train_one_epoch(
                                                        epsilon= epsilon,
                                                        delta= delta,
                                                        params_dict= params_dict
                                                    )
            
            avg_losses.append(one_epoch_metrics["avg_loss"])
            avg_len_path.append(one_epoch_metrics["avg_len_path"])
            epsilons.append(epsilon)

            tqdm.write(
                        f"Epoch {epoch}:" 
                        f"Loss = {one_epoch_metrics["avg_loss"]:.6f}, "
                        f"Avg Path = {one_epoch_metrics["avg_len_path"]:.2f}, "
                        f"Epsilon = {epsilon:.3f}, "
                        f"Delta = {delta: .4e}"
                    )

            # Update Epsilon and Delta values in the agent
            epsilon = self.epsilon_schedule.step()
            delta = self.delta_schedule.step()
            
            # Update Delta value in the environment
            self.environment.set_delta_current(delta)

            print(f'[INFO] Epsilon value updated : {epsilon}')
            print(f'[INFO] Delta value updated in agent : {delta}')
            print(f'[INFO] Delta value updated in environment : {self.environment.get_delta_current()}')
        
        # save the model checkpoints
        if epoch % saving_freq == 0:
            save_model_checkpoints(self.name, epoch, self.evaluation_q_network, self.target_q_network)

        save_last_models(
                            self.name, 
                            self.evaluation_q_network, 
                            self.target_q_network
                            )

        # Save the records of average losses, length of path and epsilon
        save_Data(self.name, avg_losses, avg_len_path, epsilons)

        # Plot the convergence of the algorithm
        plot_Convergence(self.name, avg_losses, avg_len_path)

        # The NN trained is a function that must find the optimal policy
        return self.evaluation_q_network
        