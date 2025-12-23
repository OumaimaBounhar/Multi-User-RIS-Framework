import numpy as np 
from config.parameters import Parameters
from dataset.probability import Probability
from dataset.monteCarlo import Dataset_probability
from reinforcement_learning.states import State
from typing import Tuple


class Environment():
    """ Environment class for the RL problem"""
    def __init__(   self, 
                    states : State,
                    parameters: Parameters,
                    probability : Probability, 
                    dataset_train: Dataset_probability, 
                    dataset_test: Dataset_probability
                ):
        
        ## Dataset
        self.dataset_train = dataset_train.get_Data()
        self.dataset_test = dataset_test.get_Data()
        
        ## For the Q-Learning
        self.state_space = states
        
        ## For the channel model
        self.parameters = parameters
        self.probability = probability
        
        # Set the prior at the initial state
        self.prior = self.state_space.get_state_from_index(state_index=0)
        self.ordered_list = np.arange(0,parameters.get_codebook_parameters()[0][0])
        self.List_Samples:list = [] ## To store the samples
        self.size_states = len(self.prior) # size of the vectors representing the states
        
        self.len_window_action = parameters.len_window_action
        self.len_window_channel = parameters.len_window_channel
        
        self.delta = parameters.delta_init ## Initial delta
        self.delta_decay = parameters.delta_decay ##  delta decay rate
        
    def reset_prior(self):
        ## the prior is set to the initial one (index = 0)
        self.prior = self.state_space.get_state_from_index(0)
        self.posterior = self.state_space.get_state_from_index(0)
        self.List_Samples = [] ## resets the list of samples
        
    def reset_curse_dimension(self):
        ## the prior is set to the initial one (index = 0)
        self.prior = self.posterior
        self.List_Samples = [] ## resets the list of samples
    
    def step(self, index_channel:Tuple[int,int],best_action,train_or_test:bool=True, model_type = 'DQN'):
        """" 
        Args :
        -------
            best_action_index : The index of the action chosen by the epsilon-greedy policy
            
        Returns :
        ---------
            closest_state_index : The next state index
            reward : The reward
        """
        ## Feedback for a channel stored in the dataset
        index_class_channel,index_specific_channel = index_channel
        
        if train_or_test:
            Feedback_channel = (self.dataset_train)[index_class_channel][1][index_specific_channel]
        else:
            Feedback_channel = (self.dataset_test)[index_class_channel][1][index_specific_channel]
        
        ## Loop on the size of the number of action we take
        for window in range(0,self.len_window_action):
            # codeword_to_test = best_action[window]
            codeword_to_test = best_action #Just to make the dql work for the moment as best_action is an index and not a list
            Feedback_action = Feedback_channel[codeword_to_test]
            (self.List_Samples).append((Feedback_action, codeword_to_test)) ## Stores the new sample
            
        posterior = self.probability.update(self.prior, self.ordered_list, new_sample = self.List_Samples)[0]
        #print(posterior)
        self.posterior = posterior
        
        closest_state_index:int = -1
        if model_type.upper() == 'QL':
            ## Find the closest state to the posterior probability
            min_distance = float('inf')
            for next_state_index in range(self.state_space.get_n_states()):
                next_state = self.state_space.get_state_from_index(next_state_index)
                ## Calculate the distance between posterior and the next state
                distance = float(np.linalg.norm(posterior - next_state))
                if distance < min_distance:
                    min_distance = distance
                    closest_state_index = next_state_index
            
            next_state = self.state_space.get_state_from_index(closest_state_index)
            
            if max(next_state)>=1-self.delta:
                reward = 0
            #if next_state[index_class_channel]>= 1-self.delta:
                #reward = 0
            else : 
                reward = -1
            
            return closest_state_index, reward
        
        if model_type.upper() == 'DQN':
            if max(posterior)>=1-self.delta:
                reward = 0
            #if next_state[index_class_channel]>= 1-self.delta:
                #reward = 0
            else : 
                reward = -1
            return posterior, reward
            
            ## Find the closest state to the posterior probability
            #min_distance = float('inf')
            #for next_state_index in range(self.state_space.get_n_states()):
                #next_state = self.state_space.get_state_from_index(next_state_index)
                ## Calculate the distance between posterior and the next state
                #distance = float(np.linalg.norm(posterior - next_state))
                #if distance < min_distance:
                    #min_distance = distance
                    #closest_state_index = next_state_index
            
            #next_state = self.state_space.get_state_from_index(closest_state_index)
            
            #return closest_state_index, reward
        
        
        ## Stopping condition at a terminal state
        ## Terminal state is index_class_channel+1 (1 is for the initial state see the way State is defined)
        ## terminal_state = self.state_space.get_state_from_index(index_class_channel+1)
        
            
        #print(self.prior)
        #print(posterior)
        #print(next_state)
        #print(terminal_state)
        #self.prior = posterior ## Stores the new probability
        #self.ordered_list = new_ordered_list

    def get_len_window_action(self):
        return self.len_window_action
    
    def get_len_window_channel(self):
        return self.len_window_channel
    
    def get_dataset(self):
        return self.dataset_train,self.dataset_test
    
    def get_state_space(self):
        return self.state_space
    
    def get_size_states(self):
        return self.size_states
    
    def get_delta_final(self):
        return self.state_space.get_delta()
    
    def set_delta_current(self,delta):
        self.delta = delta
        
    def get_delta_current(self):
        return self.delta
    
    def get_posterior(self):
        return self.posterior
    
    def get_size_codebook(self):
        return self.parameters.get_codebook_parameters()[0]
     