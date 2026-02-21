import numpy as np 
import random
import pandas as pd
from tqdm import tqdm
from typing import List
from matplotlib import pyplot as plt 

from reinforcement_learning.env import Environment
from config.parameters import Parameters
from reinforcement_learning.q_learning.utils import computes_percentage_unvisited_states, plot_Convergence,  save_policy, save_Q_matrix
from reinforcement_learning.deep_q_learning.components.schedules import multiplicativeDecaySchedule, LinearDecaySchedule

class QLearningAgent():
    """
    Class for the agent of Q-Learning
    """
    def __init__(   self, 
                    environment: Environment,
                    parameters: Parameters,
                    name_file:str="Data/Example_0/Q_matrices") :
        
        ## ---- The environment ----
        self.environment = environment
        self.parameters = parameters
        self.state_space = environment.get_state_space()
        self.n_states = environment.state_space.get_n_states()
        self.n_codebook_pilots = (environment.get_size_codebook())[1]
        print(f'Size of the codebook : {environment.get_size_codebook()}')
        self.n_actions = self.n_codebook_pilots
        
        ## ---- Hyperparameters ----
        self.delta_final = environment.get_delta_final() ## Final degree of precision we want to reach, should correspond to the one in states
        params_dict = parameters.get_q_learning_parameters()
        self.n_episodes = params_dict["n_episodes"]
        self.n_channels_train = params_dict["n_channels_train"]
        self.gamma = params_dict["gamma"]
        initial_q_value = params_dict["initial_q_value"]
        learning_rate_init  = params_dict["learning_rate_init"]
        learning_rate_decay = params_dict["learning_rate_decay"]
        learning_rate_min   = params_dict["learning_rate_min"]
        epsilon_init = params_dict["epsilon_init"]
        epsilon_decay = params_dict["epsilon_decay"]
        epsilon_min = params_dict["epsilon_min"]
        delta_init = params_dict["delta_init"]
        delta_decay = params_dict["delta_decay"]
        delta_min = params_dict["delta_min"] # Final degree of precision we want to reach


        ## ---- Dataset ----
        self.name = name_file
        self.dataset_train,self.dataset_test = environment.get_dataset()
        
        ## ---- Q-Matrix ----
        self.initial_value_Q_matrix = initial_q_value
        self.Q_matrix = initial_q_value * np.ones((self.n_states, self.n_actions))

        ## To check how many times the states were visited, we store counts in a "Q-matrix_frequency"
        self.Q_matrix_freq = np.zeros(self.n_states)
        self.number_update = 0
        
        # ---- Schedules ----
        self.learning_rate_schedule = LinearDecaySchedule(
                                                            init_value = learning_rate_init,
                                                            min_value = learning_rate_min,
                                                            total_steps = self.n_episodes
                                                            )
        
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
                
        ## ---- Policy ----
        self.policy = np.zeros(self.n_states, dtype = int)

    def update_Q_matrix(self, reward: float, current_state: int, next_state: int, action: int, is_terminal: bool = False) -> None:
        """Updates the Q-matrix using the Bellman equation : 
            Q(s,a) <- (1 - alpha) * Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a'))
        Args:
            reward (int): the reward received after taking the action
            current_state (int): the current state index
            next_state (int): the next state index
            action (int): the action taken
            is_terminal (bool) : If a terminal state is reached
        Returns:
            None
        """
        ## Current hyperparameters
        ## Schedule is the only source of truth value
        alpha = self.learning_rate_schedule.get()
        gamma = self.gamma

        # ---- Safety checks ----
        assert 0 <= current_state < self.n_states
        assert 0 <= next_state < self.n_states
        assert 0 <= action < self.n_actions
        assert 0.0 <= gamma <= 1.0
        assert 0.0 < alpha <= 1.0       

        ## Update the old Q-value
        old_Q_value = self.Q_matrix[current_state,action] 

        ## Record the visited states
        self.Q_matrix_freq[current_state] += 1 
        self.number_update += 1

        ## TD target
        if is_terminal:
            target = reward
        else :
            target = reward + gamma * np.max(self.Q_matrix[next_state])
        
        # TD update
        self.Q_matrix[current_state,action] = (1 - alpha) * old_Q_value + alpha * target 

    def choose_action(self, state_index: int, epsilon: float) -> int:
        """
        Choose an action following Epsilon Greedy Policy.
        =======
        Args:
        =======
        @ state_index : The index of the current state in the state space
        @ epsilon : Epsilon schedule

        Returns:
        The index of the action to choose next
        """
        assert 0 <= state_index < self.n_states
        assert 0.0 <= epsilon <= 1.0

        if random.random() > epsilon:
            q_values = self.Q_matrix[state_index]
            max_q = np.max(q_values) ## np.argmax returns the first index even if many q-values are equal (initially...)
            best_actions = np.flatnonzero(np.isclose(q_values, max_q)) ## If we have many q_values equal at the
            action_index = int(np.random.choice(best_actions)) ## To avoid selecting the
        else:
            action_index = int(np.random.randint(self.n_actions))
        
        return action_index
        
    def train(self):
        """
        Train the Q-Learning agent.
        
        """
        print("[INFO] Q-Learning Training : process Initiated...")
        print(f'The action-state space is of size {self.n_states * self.n_actions}')
        
        avg_len_train_epoch = []
        avg_len_test_epoch = []

        count_ep = 0

        epsilon = self.epsilon_schedule.get()
        delta = self.delta_schedule.get()
        learning_rate = self.learning_rate_schedule.get()

        for episode in tqdm(range(self.n_episodes)):
            
            ## Computes the number of states we visited, and if we visited them all change epsilon to use a greedy method
            percentage_unvisited_states = computes_percentage_unvisited_states()
            
            if percentage_unvisited_states == 0:
                print(f'[INFO] Q-Learning exploration : All states have been visited ! Setting epsilon to epsilon_min = {self.epsilon_schedule.reset()} ...')
                if count_ep == 10 :
                    epsilon = self.epsilon_schedule.reset()
                else:
                    count_ep += 1
            
            # Update Delta value in the agent
            delta = self.delta_schedule.step()

            # Update Delta value in the environment
            self.environment.set_delta_current(delta)

            ## Compute the average length on a test set
            avg_len_train = self.train_one_epoch(
                                                    epsilon = epsilon,
                                                    delta = delta,
                                                    learning_rate = learning_rate
                                                ) 
            avg_len_train_epoch.append(avg_len_train)
            
            # # Update Epsilon and Alpha values in the agent
            epsilon = self.epsilon_schedule.step()
            learning_rate = self.learning_rate_schedule.step()

            tqdm.write(f'Episode: {episode+1}, Epsilon: {epsilon:.2f}, Delta: {delta:.2f}, Alpha: {learning_rate: 2f}, Percentage of unvisited states/actions: {percentage_unvisited_states:.2f}')
            tqdm.write('---------------------------------------------------')
            
            if episode % self.parameters.saving_freq_QL == 0:
                
                # Save the Q-matrix
                save_Q_matrix(episode,self.name)

                # Extract policy
                for s in range(self.n_states):
                    # Choose the action with the highest Q-value for the current state 
                    best_action_index = np.argmax(self.Q_matrix[s])

                    # Update the policy with the chosen action for the current state
                    self.policy[s] = best_action_index

                # Save the Policy
                save_policy(episode,self.name)
        
        print("[INFO] Q-Learning Training : Process Completed !")
        
        # Save the Q-matrix and Policy for the last episode
        save_policy(self.n_episodes,self.name)
        save_Q_matrix(self.n_episodes,self.name)
        
        filename_frequency = self.name + f"/frequency_after_{self.n_episodes}episodes_with_delta_=_{self.parameters.delta_init}_and_alpha=_{self.parameters.learning_rate_init}.csv"
        np.savetxt(filename_frequency, self.Q_matrix_freq/self.number_update*100, delimiter=",")
        print(f"[INFO] Frequency matrix saved to {filename_frequency}")
        
        smoothed_avg_len = np.convolve(avg_len_train_epoch, np.ones(10)/10, mode='valid')
        
        plot_Convergence(self.name, smoothed_avg_len)
        
        output_df = pd.DataFrame({"Len Train": avg_len_train_epoch})
        filename_data = self.name+"/Data_len_training.dat"
        output_df.to_csv(filename_data)
        print(f"[INFO] Mean len path during QL - Training saved to {self.name}")
        
    def train_one_epoch(self, epsilon: float, delta: float, learning_rate: float):
        
        
        all_len_path = []
        
        for channel_realization in range(self.n_channels_train):
            
            current_state_index = 0 ## We always start at the Initial State
            length_path = 0 ## The length of the path we take to be able to measure how efficient the policy is
            
            ## Set the prior at the initial state
            self.environment.reset_prior()
            len_window_channel = self.environment.get_len_window_channel()
            
            ## Generates a new channel (take it randomly from the dataset given in the environment)
            if self.parameters.train_or_test:
                index_class_channel = np.random.randint(0,len(self.dataset_train)) # Random class
                index_specific_channel = np.random.randint(0,len((self.dataset_train)[index_class_channel][1])) # Random channel from this class
            else:
                #index_class_channel = np.random.randint(0,len(self.dataset_test))
                #index_specific_channel = np.random.randint(0,len((self.dataset_test)[index_class_channel][1]))
                ## Test elements in the dataset one after the other
                index_class_channel = time_step%len(self.dataset_test)
                index_specific_channel = time_step//len(self.dataset_test)
                if time_step//len(self.dataset_test) >= len((self.dataset_test)[index_class_channel][1]):
                    index_specific_channel = np.random.randint(0,len((self.dataset_test)[index_class_channel][1]))
            index_channel = (index_class_channel,index_specific_channel)
        
            for time_step in range(self.parameters.n_time_steps_ql):
                
                for path in range(self.parameters.max_len_path):
                    # max_len_path is big because we want to check if the policy really works

                    # Choose an action following Epsilon greedy Policy
                    index_action = self.choose_action(current_state_index,epsilon)
                        
                    # Update the state
                    ## reset the list of samples and put prior = posterior
                    if path % len_window_channel == 0:
                        self.environment.reset_curse_dimension()
                        
                    # next_state_index , reward = self.environment.step(index_channel,tuple_action,train_or_test= self.parameters.train_or_test, model_type = 'QL')
                    next_state_index , reward = self.environment.step(index_channel,index_action,train_or_test= self.parameters.train_or_test, model_type = 'QL')
                    # Save the reward
                    if self.parameters.train_or_test:
                        length_path += reward
                    else:
                        length_path += -1
                    
                    # Update the state-action value function
                    self.update_Q_matrix(reward, current_state_index, next_state_index, index_action, is_terminal)
                    
                    current_state_index = next_state_index
                    current_state = self.environment.state_space.get_state_from_index(current_state_index)
                    
                    delta = self.environment.get_delta_current()
                    
                    if self.parameters.train_or_test:
                        if max(current_state)>=1-delta:
                            break 
                    else:
                        probability = self.environment.get_posterior()
                        if max(probability)>=1-delta:
                            break
                        
            all_len_path.append(length_path)
        mean_length_path = np.mean(all_len_path)

        return -mean_length_path
    
