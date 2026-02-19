import numpy as np 
import random
from tqdm import tqdm
from typing import List
from matplotlib import pyplot as plt 

from reinforcement_learning.env import *
from config.parameters import Parameters
from reinforcement_learning.q_learning.utils import *

class QLearningAgent():
    """
    Class for the agent of Q-Learning
    """
    def __init__(   self, 
                    environment: Environment,
                    parameters: Parameters,
                    name_file:str="Data/Example_0/Q_matrices") :
        
        ## The environment
        self.environment = environment
        self.parameters = parameters
        self.len_window_action = environment.get_len_window_action()
        self.state_space = environment.get_state_space()
        
        ## Hyperparameters for the Bellman Equation 
        params_dict = parameters.get_q_learning_parameters()
        initial_q_value = params_dict["initial_q_value"]
        
        ## Dataset
        self.dataset_train,self.dataset_test = environment.get_dataset()
        self.name = name_file
        
        ## For the Q-Matrix
        print(f'environment.get_size_codebook(): {environment.get_size_codebook()}')
        self.n_codebook_pilots = (environment.get_size_codebook())[1]
        self.n_states = environment.state_space.get_n_states()
        self.n_actions = self.n_codebook_pilots**(self.len_window_action)
        self.Q_matrix = initial_q_value * np.ones((self.n_states, self.n_actions))
        self.initial_value_Q_matrix = initial_q_value
        
        ## The policy we output
        self.policy = np.zeros(self.n_states)
        self.delta_final = environment.get_delta_final() ## Final degree of precision we want to reach, should correspond to the one in states
        
        ## Some values we store to check how many times the states were visited
        self.Q_matrix_freq = np.zeros(self.n_states)
        self.number_update = 0
        
    def update_Q_matrix(self, reward: int, current_state: int, next_state: int, action: int, params_dict = {}) -> None:
        """Updates the Q-matrix using the Bellman equation.
        Args:
            reward (int): the reward received after taking the action
            current_state (int): the current state index
            next_state (int): the next state index
            action (int): the action taken
        Returns:
            None
        """
        
        #The old Q-value
        old_Q_value = self.Q_matrix[current_state][action] # Old value that we will update
        self.Q_matrix_freq[current_state] += 1 # To check which states were visited
        self.number_update += 1
        
        # The new Q-value
        best_action_index = np.argmax(self.Q_matrix[next_state])
        
        ## To take into account the fact that the computation of the probability is not exact, 
        ## sometimes the wrong terminal state can be reached
        
        learned_value = reward + self.parameters.gamma * self.Q_matrix[next_state][best_action_index]
        
        # Update the Q-matrix of the current state and action pair
        self.Q_matrix[current_state][action] = (1 - self.parameters.learning_rate_init) * old_Q_value + self.parameters.learning_rate_init * learned_value

    def choose_action(self, state_index: int, epsilon: float):
        """
        Choose an action following Epsilon Greedy Policy.
        """
        random_number = random.random()
        if random_number > epsilon:
            action_index = int(np.argmax(self.Q_matrix[state_index, :]))
        else:
            action_index = int(np.random.choice(self.n_actions))
        
        ## tuple_action = [action_1,action_(2),...action_len_window_action]
        ## action = n_codebook_pilots**(len_window-1) * action_1 + n_codebook_pilots**(len_window-2) * action_(2)...+action_len_window_action
        ## Ex: 37 = 3*10 + 7*1
        tuple_action = []
        action = action_index
        for window in range(0,self.len_window_action):
            action_window = action//(self.n_codebook_pilots**(self.len_window_action-window-1)) 
            action = action_window%(self.n_codebook_pilots**(self.len_window_action-window-1))
            tuple_action.append(action_window)
        
        return tuple_action,action_index
        
    def train(self, params_dict = {}):
        
        """
        Train the Q-Learning agent.
        
        """
        
        n_episodes = params_dict["n_episodes"]
        epsilon_decay = params_dict["epsilon_decay"]
        epsilon_min = params_dict["epsilon_min"]
        delta_init = params_dict["delta_init"]
        delta_decay = params_dict["delta_decay"]
        saving_freq = params_dict["saving_freq"]
        learning_rate_init = params_dict["learning_rate_init"]
        learning_rate_decay  = params_dict["learning_rate_decay"]
        learning_rate_min  = params_dict["learning_rate_min"]
        
        print("[INFO] Q-Learning Training : process Initiated...")
        print(f'The action-state space is of size {self.n_states * self.n_actions}')
        
        avg_len_train_epoch = []
        avg_len_test_epoch = []
        epsilon:float = 1.0
        count_ep = 0
        delta = delta_init
        learning_rate = learning_rate_init
        for episode in tqdm(range(n_episodes)):
            
            ## Computes the number of states we visited, and if we visited them all change epsilon to use a greedy method
            percentage_unvisited_states = self.computes_percentage_unvisited_states()
            
            if percentage_unvisited_states == 0:
                print(f'[INFO] Q-Learning exploration : All states have been visited ! Setting epsilon to epsilon_min = {epsilon_min} ...')
                if count_ep == 10 :
                    epsilon = epsilon_min
                else:
                    count_ep += 1
            
            #reward_episode = []
            delta = max(delta*delta_decay,self.delta_final)
            self.environment.set_delta_current(delta)

            ## Compute the average length on a test set
            avg_len_train = self.train_one_epoch(params_dict) # max_len_path is big because we want to check if the policy really works
            
            avg_len_train_epoch.append(avg_len_train)
            # avg_len_test_epoch.append(avg_len_test)
            
            # Decay epsilon
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            
            # Decay learning rate
            learning_rate = max(learning_rate-episode/n_episodes*learning_rate+learning_rate_min, learning_rate_min)
            params_dict["learning_rate"] = learning_rate  # Update learning rate in params_dict for use in Q-matrix updates

            # tqdm.write(f'Episode: {episode+1}/{n_episodes}, Average Length on test: no test, Epsilon: {epsilon:.2f}, Delta: {delta:.2f}, Alpha: {learning_rate: 2f}, Percentage of unvisited states/actions: {percentage_unvisited_states:.2f}')
            tqdm.write(f'Episode: {episode+1}, Epsilon: {epsilon:.2f}, Delta: {delta:.2f}, Alpha: {learning_rate: 2f}, Percentage of unvisited states/actions: {percentage_unvisited_states:.2f}')
            tqdm.write('---------------------------------------------------')
            
            if episode % saving_freq == 0:
                
                # Save the Q-matrix
                self.save_Q_matrix(episode,self.name)

                # Extract policy
                for s in range(self.n_states):
                    # Choose the action with the highest Q-value for the current state 
                    best_action_index = np.argmax(self.Q_matrix[s])

                    # Update the policy with the chosen action for the current state
                    self.policy[s] = best_action_index

                # Save the Policy
                self.save_policy(episode,self.name)
        
        print("[INFO] Q-Learning Training : Process Completed !")
        
        # Save the Q-matrix and Policy for the last episode
        self.save_policy(n_episodes,self.name)
        self.save_Q_matrix(n_episodes,self.name)
        
        filename_frequency = self.name + f"/frequency_after_{n_episodes}episodes_with_delta_=_{delta_init}_and_alpha=_{learning_rate_init}.csv"
        np.savetxt(filename_frequency, self.Q_matrix_freq/self.number_update*100, delimiter=",")
        print(f"[INFO] Frequency matrix saved to {filename_frequency}")
        
        smoothed_avg_len = np.convolve(avg_len_train_epoch, np.ones(10)/10, mode='valid')
        
        # plt.plot(avg_len_test_epoch)
        plt.plot(smoothed_avg_len)
        plt.title("Convergence of Q-Learning Algorithm")
        plt.xlabel("Iteration")
        plt.ylabel("Average Len path normalized with a window size = 10")
        plt.savefig(self.name+"/convergence_q_learning_train.png")
        
        output_df = pd.DataFrame({"Len Train": avg_len_train_epoch})
        filename_data = self.name+"/Data_len_training.dat"
        output_df.to_csv(filename_data)
        print(f"[INFO] Mean len path during QL - Training saved to {self.name}")
        
    def train_one_epoch(self, params_dict = {}):
        
        n_channels_train = params_dict["n_channels_train"]
        
        all_len_path = []
        
        for channel_realization in range(n_channels_train):
            
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
                    # Choose an action following Epsilon greedy Policy
                    if self.parameters.train_or_test:
                        tuple_action, index_action = self.choose_action(current_state_index, self.parameters.epsilon)
                    else:
                        tuple_action, index_action = self.choose_action(current_state_index, 0) ## Greedy Action during the test
                        
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
                    
                    # Update the state-action value function if we train
                    if self.parameters.train_or_test:
                        self.update_Q_matrix(reward, current_state_index, next_state_index, index_action, params_dict = params_dict)
                    
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
    
