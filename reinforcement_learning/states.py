from config.parameters import Parameters
import numpy as np 
from itertools import combinations

class State():
    def __init__(self, parameters : Parameters):
        self.parameters = parameters
        self.precision = parameters.precision ## Define how many states we will have
        ## Number of terminal states = size codebook communication
        self.n_terminal_states = parameters.get_codebook_parameters()[0][0] 
        self.delta = parameters.delta_init
        self.initialize()
    
    def initialize(self):
        """Create the state space:
        1) First state with uniform probability
        2) Terminal states
        3) Other states according to a heuristic, here: [1/2,0,..,0,..,1/2,0,..,0], [1/3,0,..,1/3,..,0,..,1/3,0,..,0]
        """
        print("[INFO] Initializing the state space in progress...")
        
        state_init = [ (1/self.n_terminal_states) for _ in range(self.n_terminal_states) ]
        counter = 1 ## Number of states
        
        #Create the terminal states
        terminal_states = np.diag([1-self.delta for _ in range(self.n_terminal_states)])
        counter += self.n_terminal_states
        state_init = np.vstack((state_init, terminal_states))
        
        method1 = False
        method2 = True
        method3 = False
        
        if method1:
            print("Just initial and terminal")
            
        if method2:
            print("1/2 and 0.8, 0.7")
            ## Create the states with the combinations of 1/2 and 1/3 
            ## States like [1/2,0,..,0,..,1/2,0,..,0], [1/3,0,..,1/3,..,0,..,1/3,0,..,0]
            for precision in range(2,self.precision+1):
                combinations_list = [c for c in combinations(range(self.n_terminal_states), precision)]
                #total_number_combination = sum(1 for _ in combinations_list)
                #print(f"[INFO] With precision {precision}, the total number of combinations is {total_number_combination}..")
                
                #Create the states with the combinations of 1/2 and 1/3
                for combination in combinations_list:
                    array = np.zeros(self.n_terminal_states)
                    array[list(combination)] = 1/precision
                    state_init = np.vstack((state_init, array))
                    counter+=1

            ## Create the states with the combinations of pairs equal to 1
            ## List of value pairs
            #value_pairs = [(0.7, 0.3), (0.9, 0.1), (0.8, 0.2), (0.6, 0.4),(0.3, 0.7), (0.1, 0.9), (0.2, 0.8), (0.4, 0.6),(0.95,0.05),(0.05,0.95)]
            #value_pairs = []
            value_pairs = [(0.6, 0.4),(0.4, 0.6),(0.3, 0.7),(0.7, 0.3),(0.8,0.2),(0.2,0.8)]
            for value_1, value_2 in value_pairs:
                # Generate all combinations of indices for the given precision
                indices_combinations = combinations(range(self.n_terminal_states),2)

                for indices in indices_combinations:
                    # Create an array of zeros
                    array = np.zeros(self.n_terminal_states)
                    # Assign the values to the specified indices
                    array[indices[0]] = value_1
                    array[indices[1]] = value_2
                    # Append the array to the list of all combinations
                    state_init = np.vstack((state_init, array))
                    counter += 1
            
        if method3:
            print("hierarchical")
            Array = [np.array([1/4,1/4,1/4,1/4,0,0,0,0]),np.array([0,0,0,0,1/4,1/4,1/4,1/4]),
                     np.array([1/2,1/2,0,0,0,0,0,0]),np.array([0,0,1/2,1/2,0,0,0,0]),
                     np.array([0,0,0,0,1/2,1/2,0,0]),np.array([0,0,0,0,0,0,1/2,1/2])]
            
            for a in range(0,len(Array)):
                state_init = np.vstack((state_init, Array[a]))
                counter += 1
            #len_vector = self.n_terminal_states
            #while len_vector//2 != 1:
                #len_vector = len_vector//2
                #array = np.zeros(self.n_terminal_states)
                #for l in range(len_vector):
                    #array[l] = 1/len_vector

        #print(states_space)
        self.n_states = counter
        print(f"[INFO] The size of the state space is {counter}..")
        self.states_space = state_init

    def get_n_states(self):
        return self.n_states
    
    def get_delta(self):
        return self.delta
    
    def get_state_from_index(self, state_index:int):
        ## From a index returns the corresponding state p = [......] (vector of size codebook of communication)
        return self.states_space[state_index]
    