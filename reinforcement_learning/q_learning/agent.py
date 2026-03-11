import random
import numpy as np 
from tqdm import tqdm

from config.parameters import Parameters
from reinforcement_learning.env import Environment
from experiments.store import ExperimentPaths
from reinforcement_learning.q_learning.utils import extract_Policy, computes_percentage_unvisited_states, plot_Convergence,  save_Policy_matrix, save_Q_matrix, save_frequency_update_per_state, save_training_metrics
from reinforcement_learning.deep_q_learning.components.schedules import multiplicativeDecaySchedule, LinearDecaySchedule

class QLearningAgent():
    """
    Class for the agent of Q-Learning
    """
    def __init__(   
            self, 
            environment: Environment,
            parameters: Parameters,
            paths: ExperimentPaths
        ) :
        
        ## ---- The environment ----
        self.parameters = parameters
        self.environment = environment
        self.paths = paths

        self.state_space = environment.get_state_space()
        self.n_states = environment.state_space.get_n_states()
        self.n_codebook_pilots = (environment.get_size_codebook())[1]
        print(f'Size of the codebook : {environment.get_size_codebook()}')
        self.n_actions = self.n_codebook_pilots
        
        ## ---- Hyperparameters ----
        self.delta_final = environment.get_delta_final() ## Final degree of precision we want to reach, should correspond to the one in states
        params_dict = parameters.get_q_learning_parameters()
        
        self.saving_freq = params_dict["saving_freq"]

        self.n_episodes = params_dict["n_episodes"]
        self.n_channels_train = params_dict["n_channels_train"]
        self.n_time_steps = params_dict["n_time_steps"]
        self.max_len_path = params_dict["max_len_path"]

        self.gamma = params_dict["gamma"]
        self._greedy_mode = params_dict["_greedy_mode"]
        self.initial_q_value = params_dict["initial_q_value"]

        learning_rate_init  = params_dict["learning_rate_init"]
        learning_rate_min   = params_dict["learning_rate_min"]

        epsilon_init = params_dict["epsilon_init"]
        epsilon_decay = params_dict["epsilon_decay"]
        self.epsilon_min = params_dict["epsilon_min"]

        delta_init = params_dict["delta_init"]
        delta_decay = params_dict["delta_decay"]
        delta_min = params_dict["delta_min"] # Final degree of precision we want to reach

        ## ---- Dataset ----
        self.dataset_train = environment.get_dataset()
        
        ## ---- Q-Matrix ----
        self.Q_matrix = self.initial_q_value * np.ones((self.n_states, self.n_actions))

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
                                                            min_value = self.epsilon_min
                                                            )
        
        self.delta_schedule = multiplicativeDecaySchedule(
                                                            init_value = delta_init,
                                                            decay = delta_decay,
                                                            min_value = delta_min
                                                            )

    def update_Q_matrix(self, alpha, gamma, reward: float, current_state: int, next_state: int, action: int, is_terminal: bool = False) -> None:
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
            action_index = int(np.random.choice(best_actions)) ## To avoid selecting the same action at the beginning when all q-values are equal, we randomly select among the best actions
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
        count_ep = 0 ## Counter for the greedy switch

        epsilon = self.epsilon_schedule.get()
        delta = self.delta_schedule.get()
        learning_rate = self.learning_rate_schedule.get()

        for episode in tqdm(range(self.n_episodes)):
            
            # ---- Exploration monitoring ----
            percentage_unvisited_states = computes_percentage_unvisited_states(self.Q_matrix_freq)
            
            ## If all states were visited, wait to confirm after 10 episodes then change epsilon to epsilon min to use a greedy method
            if (percentage_unvisited_states == 0) and (not self._greedy_mode):
                count_ep += 1
                if count_ep >= 10 :
                    self.epsilon_schedule.change(self.epsilon_min)
                    epsilon = self.epsilon_schedule.get()
                    self._greedy_mode = True
                    print(
                        f"[INFO] All states visited for 10 consecutive episodes. "
                        f"Switching to greedy mode (epsilon={epsilon})."
                        )
                else:
                    count_ep = 0
            
            # ---- Update delta and apply to environment ----
            delta = self.delta_schedule.step()
            self.environment.set_delta_current(delta)

            # ---- Train one episode ----
            avg_len_train = self.train_one_episode(
                                                    epsilon = epsilon,
                                                    delta = delta,
                                                    alpha = learning_rate,
                                                    gamma = self.gamma,
                                                )
            avg_len_train_epoch.append(avg_len_train)
            
            # ---- Update epsilon and learning rate schedules ----
            epsilon = self.epsilon_schedule.step()
            learning_rate = self.learning_rate_schedule.step()

            tqdm.write(
                        f"Episode {episode+1}/{self.n_episodes} | "
                        f"Epsilon: {epsilon:.3f} | "
                        f"Delta: {delta:.3f} | "
                        f"Alpha: {learning_rate:.3f} | "
                        f"Unvisited states: {percentage_unvisited_states:.2f}%"
                    )
            
            if episode % self.saving_freq == 0:
                
                # Save the Q-matrix
                save_Q_matrix(
                    self.paths,
                    self.Q_matrix,
                    episode
                )

                # Extract policy
                policy = extract_Policy(self.Q_matrix)

                # Save policy
                save_Policy_matrix(
                    self.paths,
                    policy, 
                    episode
                )
        
        print("[INFO] Q-Learning Training : Process Completed !")
        
        # ---- Final save ----
        final_policy = extract_Policy(self.Q_matrix)
        save_Policy_matrix(
            self.paths, 
            final_policy, 
            self.n_episodes
        )

        save_Q_matrix(
            self.paths,
            self.Q_matrix, 
            self.n_episodes
        )
        
        save_frequency_update_per_state(
            self.paths, 
            self.Q_matrix_freq, 
            self.n_episodes, 
            self.delta_schedule.init_value, 
            self.learning_rate_schedule.init_value
        )
        
        
        save_training_metrics(self.paths, avg_len_train_epoch)

        smoothed_avg_len = np.convolve(avg_len_train_epoch, np.ones(10)/10, mode='valid')
        plot_Convergence(self.paths, smoothed_avg_len)
        
    def train_one_episode(self, epsilon: float, delta: float, alpha: float, gamma: float):
        """ 
        Run one training epoch
        =======
        Args:
        =======
        @ epsilon : for the Epsilon Greedy Policy
        @ delta: for the stopping criteria
        @ learning_rate: value of alpha

        Returns:
        @ -mean_length_path : The average length of path after all the channel realization
        """
        all_len_path = []
        
        for _ in range(self.n_channels_train):
                        
            # Length of the path metric to measure Policy efficiency
            length_path = 0 
            
            ## Generates a new channel (take it randomly from the dataset given in the environment)
            index_class_channel = np.random.randint(0,len(self.dataset_train)) 
            index_specific_channel = np.random.randint(0,len((self.dataset_train)[index_class_channel][1]))
            
            index_channel = (index_class_channel,index_specific_channel)
            done = False
        
            for _ in range(self.n_time_steps):

                """ ---------------------Modification during last test----------------------- """
                ## Start at the initial state
                current_state_index = 0 
            
                ## Set the prior at the initial state
                self.environment.reset_prior()
                """ ------------------------------------------------------------------------- """
                
                for _ in range(self.max_len_path):

                    # Choose an action following Epsilon greedy Policy
                    index_action = self.choose_action(
                        current_state_index,
                        epsilon
                    )
                    
                    next_state_index , reward, info = self.environment.step(
                        index_channel,
                        index_action,
                        model_type = 'QL'
                    )

                    # Save the reward
                    length_path += reward
                    
                    # Update the state-action value function
                    self.update_Q_matrix(
                        alpha, 
                        gamma, 
                        reward, 
                        current_state_index, 
                        next_state_index, 
                        index_action, 
                        is_terminal= info["Terminal_state"]
                    )
                    
                    current_state_index = next_state_index
                    current_state = self.environment.state_space.get_state_from_index(
                        current_state_index
                    )
                    
                    delta = self.environment.get_delta_current()
                    
                    if max(current_state)>=1-delta:
                        done = True
                        break 
                
                if done:
                    break

            all_len_path.append(length_path)
        mean_length_path = np.mean(all_len_path)

        return -mean_length_path
    
