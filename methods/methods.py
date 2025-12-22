import numpy as np
from typing import List
import random as r
from config.parameters import Parameters
from envs.feedback import Feedback
from envs.channel import Channel
from methods.Probability import Probability
from reinforcement_learning.q_Learning.states import *
from reinforcement_learning.deep_q_learning.deep_q_learning import *
import torch

class Methods:
    """Different methods are implemented:
    They all output the guessed best codeword of communication"""
    def __init__(self,
                parameters:Parameters,
                channel:Channel,
                feedback:Feedback,
                probability:Probability,
                state: State,
                # Policy_Q:int = 0,
                Policy_Q: np.ndarray,
                Policy_network: DQN = None
                ):
        ### For the channel ###
        self.parameters = parameters
        self.channel = channel
        self.feedback = feedback
        self.probability= probability
        self.state = state

        self.len_window_channel = parameters.len_window_channel # length of the window to take into account the evolution of the channel, we assume the channel does not change drastically in the window
        self.len_window_action = parameters.len_window_action
        self.Hierarchical_possible = check_size_cd(parameters.type_codebooks,parameters.size_codebooks) # Checks that the size of the codebooks are correct and fix it if not
        
        # Q-learning
        self.Policy_Q = Policy_Q
        
        # Deep Q-Learning
        self.evaluation_q_network = Policy_network
        
    def get_parameters(self):
        return {
            "len_window_channel": self.len_window_channel,
            "len_window_action": self.len_window_action,
            "Hierarchical_possible": self.Hierarchical_possible,
            "Policy_Q": self.Policy_Q
        }
        
    def set_Policy(self, Policy_network):
        #print(Policy_network)
        self.evaluation_q_network = Policy_network
        
    def forget(self)->None:
        """A new receiver appears, no knowledge is accessible we reinitialize all methods with no prior knowledge"""
        size_codebooks, type_codebooks = self.parameters.get_codebook_parameters()
        ### Exhaustive ###
        self.counter_exhaustive = 0 # Count the number of beams tested (we test all codeword of communications one after the other then starts again)
        self.list_exh = np.zeros(size_codebooks[0]) # Contains the RSE for the codewords tested for exhaustive search
        ### Hierarchical ###
        if self.Hierarchical_possible:
            self.K,self.M = find_K_M_hierarchical(type_codebooks[1])
            self.counter_hierarchical = 0 # Count the number of beams tested for hierarchical search (between 0 and K*M)
            self.argmax_hier = 0
            self.list_hier = np.zeros(self.M) # contains the M elements tested to compare them and select the next branch
            self.hier_best_beam = r.randint(0,size_codebooks[0]-1) # the best beam found by the hierarchical search, initialized randomly
        ### Random Sampling ###
        self.prior_random = np.ones(size_codebooks[0])/size_codebooks[0] # The prior for the random sampling method
        self.ordered_list_random = np.argsort(self.prior_random)[::-1] 
        self.counter_random = 0 # Count the number of beams tested
        self.list_random:list = []
        self.posterior_random = np.ones(size_codebooks[0])/size_codebooks[0]
        ### Q_learning Sampling ###
        self.prior_q = np.ones(size_codebooks[0])/size_codebooks[0] # The prior for the random sampling method
        self.posterior_q = np.ones(size_codebooks[0])/size_codebooks[0] 
        self.ordered_list_q = np.argsort(self.prior_q)[::-1] 
        self.counter_channel_q = 0 # Count the number of pilots tested before the channel changes too much
        self.counter_action_q = 0 # Count the number of pilots tested (an action corresponds corresponds to self.len_window_action pilots)
        self.list_q:list = []
        self.actions_q = [] # all actions that we took
        
        ### Deep_Q_learning Sampling ###
        self.prior_deep_q = np.ones(size_codebooks[0])/size_codebooks[0] # The prior for the random sampling method
        self.posterior_deep_q = np.ones(size_codebooks[0])/size_codebooks[0] 
        self.ordered_list_deep_q = np.argsort(self.prior_q)[::-1] 
        self.counter_channel_deep_q = 0 # Count the number of pilots tested before the channel changes too much
        self.counter_action_deep_q = 0 # Count the number of pilots tested (an action corresponds corresponds to self.len_window_action pilots)
        self.list_deep_q:list = []
        
    def optimal_codeword(self)->int:
        List_RSE = []
        ### To find the real optimal codeword ###
        ### Exhaustive search on the communication codebook (codebook_used=0), no noise ###
        size_codebooks, type_codebooks = self.parameters.get_codebook_parameters()
        for M in range(size_codebooks[0]):
            self.feedback.transmit(M,codebook_used=0)
            RSE = self.feedback.get_feedback(noise = False)
            List_RSE.append(RSE)
        argmax_RSE = int(np.argmax(List_RSE)) ## True Best codeword
        return argmax_RSE
    
    def exhaustive(self)->int:
        ### Exhaustive search on the communication codebook (codebook_used=0), noise ###
        ### Test beams one after the other and declare the index of the highest RSE of the list ###
        size_codebooks, type_codebooks = self.parameters.get_codebook_parameters()
        count = self.counter_exhaustive
        if count == size_codebooks[0]:
            count = 0
            self.counter_exhaustive = 0
        self.feedback.transmit(count,codebook_used=0)
        RSE = self.feedback.get_feedback(noise = True)
        self.list_exh[count] = RSE
        argmax_RSE = int(np.argmax(self.list_exh)) ## Best codeword in the list
        count += 1
        self.counter_exhaustive = count
        return argmax_RSE
    
    def hierarchical(self)->int:
        ### Hierarchical search, noise ###
        count = self.counter_hierarchical # Keeps track of how many codewords have been tested so far
        K_hier, M_hier = self.K,self.M # K = total levels, M = number of branches per level
        argmax_hier = self.argmax_hier # Stores the best subgroup found so far
        
        k = count//M_hier # length of the tree we reached
        n = count%M_hier ## n is the number of the leaf tested in the tree at length k

        ## Pilot sent ##
        index_beam = n + argmax_hier*M_hier + sum([M_hier**k_index for k_index in range(1,k+1)]) # index of the pilot sent n is one of the M leaves we test, M*argmaxhier is the index of the best codeword found at the parent node, and sum is to take into account our codebook

        self.feedback.transmit(index_beam,codebook_used=1)
        RSE = self.feedback.get_feedback(noise=True)
        self.list_hier[count%M_hier] = RSE

        count += 1
        self.counter_hierarchical = count
        if count%M_hier == 0: ## If search is over at the k level of the tree (M codewords were tested), then test next level
            argmax_hier = argmax_hier*M_hier + int(np.argmax(self.list_hier))
            self.argmax_hier = argmax_hier
            for elmt in range(0,M_hier):
                self.list_hier[elmt] = 0
            
        ## If search is over declare the codeword found
        # If search is over reset the counter and start again
        if count == K_hier*M_hier:
            self.hier_best_beam = argmax_hier
            self.argmax_hier = 0
            self.counter_hierarchical = 0
            
        return self.hier_best_beam
    
    def random_sampling(self)->int:
        ### Randomly selects pilots, then compute probability, with noise ###
        size_codebooks, type_codebooks = self.parameters.get_codebook_parameters()
        prior = self.prior_random
        ordered_list = self.ordered_list_random
        counter = self.counter_random
        Samples = self.list_random 
            
        ### Pilot sent ###
        index_codeword_tested = r.randint(0, size_codebooks[1]-1)
            
        self.feedback.transmit(index_codeword_tested,codebook_used=1)
        RSE = self.feedback.get_feedback(noise = True)
        Samples.append((RSE,index_codeword_tested))
        self.list_random = Samples
        
        ### Update Probability ###
        #print(ordered_list)
        #print(prior)
        prior,ordered_list = self.probability.update(prior,ordered_list,Samples)
        self.ordered_list_random = ordered_list
        self.posterior_random = prior
        
        best_codeword = ordered_list[0]

        counter += 1
        self.counter_random = counter
        
        # if the channel changed too much we update the proba
        if counter == self.len_window_channel:
            prior,ordered_list = self.probability.update_proba_channel(prior)
            self.prior_random = prior
            self.ordered_list_random = ordered_list
            counter = 0
            self.counter_random = counter
            Samples = []
            self.list_random = Samples
            
        return best_codeword 
    
    def q_learning_sampling(self)->int:
        ###  Selects pilots with Q-learning policy, then compute probability, with noise ###
        size_codebooks, type_codebooks = self.parameters.get_codebook_parameters()
        prior = self.prior_q
        ordered_list = self.ordered_list_q
        counter_channel = self.counter_channel_q
        Samples = self.list_q
        all_actions = self.actions_q
        
        current_proba = self.posterior_q # To make the decision about the next codeword to test
        
        ### Pilot sent ###
        ## Find the closest state to the posterior probability
        min_distance = float('inf') #change
        closest_state_index = -1
        for next_state_index in range(self.state.get_n_states()):
            next_state = self.state.get_state_from_index(next_state_index)
            # Calculate the distance between posterior and the next state
            distance:float = float(np.linalg.norm(current_proba - next_state))
            if distance < min_distance:
                min_distance = distance
                closest_state_index = next_state_index
        
        best_action = int(self.Policy_Q[closest_state_index])
        if best_action in all_actions:
            best_action = r.randint(0,size_codebooks[1]-1)
        
        # print(len(self.Policy_Q))
        # print(best_action) 
        ####### Peut être plus efficace en stockant best action ######
        self.actions_q.append(best_action)
        tuple_action = []
        action = best_action
        for window in range(0,self.len_window_action):
            action_window = action//(size_codebooks[1]**(self.len_window_action-window-1)) 
            action = action_window%(size_codebooks[1]**(self.len_window_action-window-1))
            tuple_action.append(action_window)
            
        self.counter_action_q = self.counter_action_q + 1
        index_codeword_tested = tuple_action[self.counter_action_q-1]
        if self.counter_action_q == self.len_window_action:
            self.counter_action_q = 0
        
        self.feedback.transmit(index_codeword_tested,codebook_used=1)
        RSE = self.feedback.get_feedback(noise = True)
        Samples.append((RSE,index_codeword_tested))
        self.list_q = Samples
        
        ### Update Probability ###
        #print(ordered_list)
        #print("before")
        #print(prior)
        
        prior,ordered_list = self.probability.update(prior,ordered_list,Samples)
        self.posterior_q = prior
        
        #print(prior)
        #print(Samples)
        best_codeword = ordered_list[0]
        
        #print(prior)
        
        counter_channel += 1
        self.counter_q = counter_channel
        # if the channel changed too much we update the proba
        if counter_channel == self.len_window_channel:
            prior,ordered_list = self.probability.update_proba_channel(prior)
            self.prior_q = prior
            self.ordered_list_q = ordered_list
            counter_channel = 0
            self.counter_q = counter_channel
            Samples = []
            self.list_q = Samples
            
        return best_codeword
    
    def deep_q_learning_sampling(self)->int:
        ###  Selects pilots with Q-learning policy, then compute probability, with noise ###
        size_codebooks, type_codebooks = self.parameters.get_codebook_parameters()
        prior = self.prior_deep_q
        ordered_list = self.ordered_list_deep_q
        counter_channel = self.counter_channel_deep_q
        Samples = self.list_deep_q
        
        current_proba = self.posterior_deep_q # To make the decision about the next codeword to test
        
        ### Pilot sent ###
        state_tensor = torch.tensor(current_proba, dtype= torch.float32).to(self.evaluation_q_network.device)
        q_values = self.evaluation_q_network(state_tensor)
        #print("proba")
        #print(current_proba)
        # print(q_values)
        # print(len(q_values))
        best_action = torch.argmax(q_values).item()
        # print(f'best_action : {best_action}')
                
        #best_action = int(self.Policy[closest_state_index])
        ####### Peut être plus efficace en stockant best action ######
    
        tuple_action = []
        action = best_action
        for window in range(0,self.len_window_action):
            action_window = action//(size_codebooks[1]**(self.len_window_action-window-1)) 
            action = action_window%(size_codebooks[1]**(self.len_window_action-window-1))
            tuple_action.append(action_window)
            
        self.counter_action_deep_q = self.counter_action_deep_q + 1
        index_codeword_tested = tuple_action[self.counter_action_deep_q-1]
        if self.counter_action_deep_q == self.len_window_action:
            self.counter_action_deep_q = 0
        
        self.feedback.transmit(index_codeword_tested,codebook_used=1)
        RSE = self.feedback.get_feedback(noise = True)
        Samples.append((RSE,index_codeword_tested))
        self.list_deep_q = Samples
        
        ### Update Probability ###
        #print(ordered_list)
        #print("before")
        #print(prior)
        
        prior,ordered_list = self.probability.update(prior,ordered_list,Samples)
        self.posterior_deep_q = prior
        
        #print(prior)
        #print(Samples)
        best_codeword = ordered_list[0]
        
        #print(prior)
        
        counter_channel += 1
        self.counter_deep_q = counter_channel
        
        # if the channel changed too much we update the proba
        if counter_channel == self.len_window_channel:
            prior,ordered_list = self.probability.update_proba_channel(prior)
            self.prior_deep_q = prior
            self.ordered_list_deep_q = ordered_list
            counter_channel = 0
            self.counter_deep_q = counter_channel
            Samples = []
            self.list_deep_q = Samples
            
        return best_codeword 
    
    
    def test_narrow(self,correct_class)->int:
        ### Randomly selects pilots, then compute probability, with noise ###
        size_codebooks, type_codebooks = self.parameters.get_codebook_parameters()
        
        proba = self.posterior_q
        ordered_list = np.argsort(proba)[::-1]
        Samples = []
        
        ### Pilot sent ###
        for best in range(0,2):
            #index_codeword_tested = correct_class + size_codebooks[1] - size_codebooks[0]
            index_codeword_tested = ordered_list[best] + size_codebooks[1] - size_codebooks[0]
            self.feedback.transmit(index_codeword_tested,codebook_used=1)
            RSE = self.feedback.get_feedback(noise = True)
            Samples.append((RSE,index_codeword_tested))
            
            #index_codeword_tested = r.randint(0, size_codebooks[0]-1) + size_codebooks[1] - size_codebooks[0]
  
        prior = np.ones(size_codebooks[0])/size_codebooks[0] # The prior for the random sampling method
        ordered_list = np.argsort(self.prior_random)[::-1] 
        
        ### Update Probability ###
        
        prior,ordered_list = self.probability.update(prior,ordered_list,Samples)
        best_codeword = ordered_list[0]
        #print(prior)
            
        return best_codeword 
    
    def test_narrow_2(self,correct_class)->int:
        ### Randomly selects pilots, then compute probability, with noise ###
        size_codebooks, type_codebooks = self.parameters.get_codebook_parameters()
        
        proba = self.posterior_q
        ordered_list = np.argsort(proba)[::-1]
        Samples = []
        
        ### Pilot sent ###
        best_codeword = 0
        best_RSE = 0
        for best in range(0,2):
            #index_codeword_tested = correct_class + size_codebooks[1] - size_codebooks[0]
            index_codeword_tested = ordered_list[best] + size_codebooks[1] - size_codebooks[0]
            self.feedback.transmit(index_codeword_tested,codebook_used=1)
            RSE = self.feedback.get_feedback(noise = True)
            if best_RSE < RSE:
                best_RSE = RSE
                best_codeword = index_codeword_tested - size_codebooks[1] + size_codebooks[0]
            
        return best_codeword 
    
    
    def test_maxprob(self,correct_class):
        ### Randomly selects pilots, then compute probability, with noise ###
        size_codebooks, type_codebooks = self.parameters.get_codebook_parameters()
        
        proba = self.posterior_q
        ordered_list = np.argsort(proba)[::-1]
        #print(proba)
        Sum = 0
        for best in range(0,1):
            Sum = Sum + proba[ordered_list[best]]
            #print(Sum)
            
        return Sum