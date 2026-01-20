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
from reinforcement_learning.deep_q_learning.utils import save_model_checkpoints, plot_Convergence, save_Data


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
        epsilon_init = params_dict["epsilon_init"],
        epsilon_decay = params_dict["epsilon_decay"]
        epsilon_min = params_dict["epsilon_min"]
        delta_init = params_dict["delta_init"],
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
        
        # Initialization of delta value controlled by the environment for the stopping criteria
        self.environment.delta = self.delta_schedule.get()

        # ---- Policy  ----
        self.policy = np.zeros([self.n_states, self.n_actions])
        
    def choose_action(self, state):
        """ Choose an action following the Epsilon Greedy Policy"""

        # Initialization of epsilon value
        epsilon = self.epsilon_schedule.get()

        if np.random.random() < epsilon :
            action = np.random.randint(self.n_actions)
        else :
            state_tensor = torch.tensor(state, dtype= torch.float32).to(self.evaluation_q_network.device)
            with torch.no_grad():
                q_values = self.evaluation_q_network(state_tensor)
                action = torch.argmax(q_values).item()
        return action
    
    def train(self, params_dict = {}, testing_objects_dict = {}):
        """" Train the Deep Q-Learning Network """
        
        batch_size = params_dict["batch_size"]
        n_epochs = params_dict["n_epochs"]
        n_time_steps = params_dict["n_time_steps"]
        max_len_path = params_dict["max_len_path"]
        train_or_test = params_dict["train_or_test"]
        saving_freq = params_dict["saving_freq"]
        test_freq = params_dict["test_freq"]
        n_channels_train = params_dict["n_channels_train"]
        freq_update_target = params_dict["freq_update_target"]
        tau = params_dict["tau"]
        do_gradient_clipping = params_dict["do_gradient_clipping"]
        gamma = params_dict["gamma"]
        targetNet_update_method = params_dict["targetNet_update_method"]
        max_norm = params_dict["max_norm"]
        
        print(f"Initialize Training for DQN Network on Device : {self.target_q_network.device}")
        
        epsilons = []
        avg_losses = []
        avg_len_path = []
        for epoch in tqdm(range(n_epochs)):
            
            all_len_path = []
            losses = []
            
            for channel_realization in range(n_channels_train):
                
                current_state = self.environment.state_space.get_state_from_index(0) ## We always start at the Initial State
                length_path = 0 ## The length of the path we take to be able to measure how efficient the policy is
                
                ## Set the prior at the initial state
                self.environment.reset_prior()
                len_window_channel = self.environment.get_len_window_channel()
                
                ## Generates a new channel (take it randomly from the dataset given in the environment)
                if train_or_test:
                    index_class_channel = np.random.randint(0,len(self.dataset_train)) # Random class
                    index_specific_channel = np.random.randint(0,len((self.dataset_train)[index_class_channel][1])) # Random channel from this class
                else:
                    ## Test elements in the dataset one after the other
                    index_class_channel = time_step%len(self.dataset_test)
                    index_specific_channel = time_step//len(self.dataset_test)
                    if time_step//len(self.dataset_test) >= len((self.dataset_test)[index_class_channel][1]):
                        index_specific_channel = np.random.randint(0,len((self.dataset_test)[index_class_channel][1]))
                index_channel = (index_class_channel,index_specific_channel)
                
                for time_step in range(n_time_steps):
                    
                    for path in range(max_len_path):
                        if train_or_test:
                            index_action = self.choose_action(current_state, epsilon)
                        else:
                            index_action = self.choose_action(current_state, 0) ## Greedy Action during the test
                        
                        # Update the state
                        ## reset the list of samples and put prior = posterior
                        if path % len_window_channel == 0:
                            self.environment.reset_curse_dimension()
                            
                        next_state, reward = self.environment.step(index_channel,index_action,train_or_test=train_or_test, model_type = 'DQN')
                        
                        # Save the reward
                        if train_or_test:
                            length_path += reward
                        else:
                            length_path += -1
                            
                        # # Store the transition in the Replay Buffer if we train
                        if train_or_test:
                            self.replay_buffer.store_transition(current_state, index_action, reward, next_state)
                            
                        current_state = next_state
                        
                        # Stop condition in a terminal state
                        delta = self.environment.delta
                        print(f'delta value in env = {self.environment.get_delta_current()}')
                        print(f'delta value in agent = {delta}')
                        
                        if train_or_test:
                            #print(current_state)
                            if max(current_state)>=1-delta:
                                break 
                        else:
                            probability = self.environment.get_posterior()
                            if max(probability)>=1-delta:
                                break
                            
                    all_len_path.append(1-length_path)
                    
                    #print(self.replay_buffer.memory_counter)
                    if self.replay_buffer.memory_counter >= batch_size:
                        # Sample a batch from the Replay Buffer
                        current_state_batch, action_batch, reward_batch, next_state_batch = self.replay_buffer.sample_buffer()
                        
                        # Start the training process
                        loss_value = dqn_learn_update_step(self.evaluation_q_network, self.target_q_network, self.optimizer, current_state_batch, action_batch, reward_batch, next_state_batch, gamma, do_gradient_clipping, max_norm )
                        losses.append(loss_value)
                        
            # save the model checkpoints
            if epoch % saving_freq == 0:
                save_model_checkpoints()
            
            avg_losses.append(np.mean(losses) if len(losses) > 0 else np.nan)
            avg_len_path.append(np.mean(all_len_path))
            
            tqdm.write(f"Epoch {epoch}: Loss = {np.mean(losses):.6f}, Avg Path = {np.mean(all_len_path):.2f}, Epsilon = {epsilon:.3f}")

            # update the epsilon value
            epsilons.append(epsilon)
            epsilon = epsilon.step()
            
            # Update delta
            self.environment.delta = self.delta_schedule.step()
            print(f'[INFO] Delta value : f{delta}')
            
            # Update target network every `target_update_freq` epochs
            if epoch % freq_update_target == 0:
                dqn_target_update(self.evaluation_q_network, self.target_q_network, tau, targetNet_update_method)
        
        torch.save(self.evaluation_q_network.state_dict(), checkpoint_dir + 'last_eval.pth')
        torch.save(self.target_q_network.state_dict(), checkpoint_dir + 'last_target.pth')
        print("[INFO] Deep Q-Learning Training : Process Completed !")

        save_Data(self.name, avg_losses, avg_len_path, epsilons)

        # Plot the convergence of the algorithm
        plot_Convergence(self.name, avg_losses, avg_len_path)

        Policy = self.evaluation_q_network # The NN trained is a function that must find the optimal policy

        return Policy
        