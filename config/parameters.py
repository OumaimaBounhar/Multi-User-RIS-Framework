from typing import List
from typing import Tuple
import math
import numpy as np


class Parameters :
    """ Contains all of the parameters for our simulation:
    For the channel: N_BS = number of antennas of the BS, N_UE = Number of users or number of antennas of the user, N_RIS = Number of reflective elements on the RIS,
    type_channel is "IID" => with parameters mean_channel and std_channel /or "half-spaced ULAs" => with parameters paths and lambda
    For the noise: mean_noise and std_noise
    For the symbols sent: type_modulation
    For the codebooks: One codebook is for the communication phase, the other one is for the pilots sent: type_codebooks is a list of 2 elements, both of them can be "DFT" or "random"
    For DQN/Q-Learning: Various hyperparameters like gamma, learning_rate, etc.
    """
    def __init__(   self, 
                    N_R:int = 64, 
                    N_T:int = 1, 
                    N_RIS:int = 100, 
                    size_codebooks:List[int] = [16,30], 
                    type_codebooks:List[str] = ["Narrow_8","Hierarchical_3_2"],
                    mean_noise:float = 0,
                    SNR: int = 30,
                    snr_values:List[int] = [-100,-50,0],
                    modification_channel: int = 0,
                    type_channel:str = "half-spaced ULAs",
                    type_modulation:str = "BPSK", 
                    mean_channel:float=0, 
                    std_channel:List[float] = [], 
                    sigma_alpha = 0, 
                    params_list: List[int] = [32, 64],
                    batch_size: int = 128,
                    replay_buffer_memory_size: int = None,
                    loss_fct: str = 'mse',
                    gamma: float = 0.99,
                    n_epochs: int = 10000,
                    n_time_steps_dqn: int = 200,
                    n_channels_train_DQN: int = 10,
                    freq_update_target: int = 1000,
                    tau: float = 0.05,
                    max_norm: float =1,
                    do_gradient_clipping: bool = True,
                    epsilon_init: float = 1.0,
                    epsilon_decay: float = 0.999,
                    epsilon_min: float = 0.01,
                    delta_init: float = 1e-1,
                    delta_decay: float = 1,
                    delta_min: float = 5 * 1e-2, 
                    targetNet_update_method : str = "soft",
                    Train_Deep_Q_Learning: bool = True,
                    saving_freq_DQN: int = 1,
                    test_freq_DQN: int = 1,
                    n_episodes: int = 20,
                    n_time_steps_ql: int = 100,
                    n_channels_train_QL: int = 10,
                    max_len_path: int = 20,
                    len_path: int = 10,
                    learning_rate_init: float = 5e-4,
                    learning_rate_decay: float = 0.99,
                    learning_rate_min: float = 1e-4,
                    Train_Q_Learning: bool = False,
                    saving_freq_QL: int = 1,
                    test_freq_QL: int = 1,
                    precision: int = 2,  
                    blabla_other_states: int = 1, 
                    min_representatives_q_learning_train: int = 100,  
                    min_representatives_q_learning_test: int = 10
                    # input_dims: int = None,
                    ) :
        
        ### For the channel ###
        
        self.N_R = N_R # Number antennas at the Receiver
        self.N_T = N_T # Number antennas at the Transmitter
        self.N_RIS = N_RIS  # Number Reflective elements at the RIS
        
        self.sigma_alpha = sigma_alpha # Variance of the attenuation of paths 
        self.modification_channel = modification_channel
        self.type_channel = type_channel
        
        if type_channel == "IID":
            self.mean_channel = mean_channel
            self.std_channel = std_channel
        
        if type_channel == "half-spaced ULAs":
            self.paths = [1,1,0] # Number of paths for the link Transmitter-RIS, RIS-Receiver, Transmitter-Receiver
            # self.paths = [2,2,0] # Ex 53
            # self.paths = [1,1,0] # Ex 54
            self.alpha = np.array([[[1,0.3,0.3],[sigma_alpha]*3],[[1,0.3,0.3],[sigma_alpha]*3],[[0.5,0.3,0.3],[sigma_alpha]*3]]) # Attenuation of paths [1 for the LOS, less for NLOS]
        
        ### For the noise and symbols sent ###
        self.SNR = SNR
        self.snr_values = snr_values
        self.mean_noise = mean_noise
        self.std_noise =  math.sqrt(10**(-SNR/10))
        self.type_modulation = type_modulation
        
        ### For the codebook ###
        self.size_codebooks = size_codebooks
        print(f'size_codebooks : {size_codebooks}')
        self.type_codebooks = type_codebooks # Communication / Pilots
        
        ### For Q-Learning parameters ###
        self.n_episodes = n_episodes
        self.n_channels_train_QL = n_channels_train_QL
        self.n_time_steps_ql = n_time_steps_ql
        self.len_path = len_path

        self.initial_q_value = -len_path 
        
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_min = learning_rate_min

        self.Train_Q_Learning = Train_Q_Learning
        self.saving_freq_QL= saving_freq_QL
        self.test_freq_QL = test_freq_QL
        
        self.precision = precision
        self.blabla_other_states = blabla_other_states
        self.min_representatives_q_learning_train = min_representatives_q_learning_train
        self.max_samples_q_learning_train = 2*min_representatives_q_learning_train*size_codebooks[0] 
        self.min_representatives_q_learning_test = min_representatives_q_learning_test
        self.max_samples_q_learning_test = 2*min_representatives_q_learning_test*size_codebooks[0] 

        
        ### For DQN Parameters ###
        self.params_list = params_list
        self.loss_fct = loss_fct
        self.batch_size = batch_size
        self.replay_buffer_memory_size = replay_buffer_memory_size

        self.n_epochs = n_epochs
        self.n_channels_train_DQN = n_channels_train_DQN
        self.n_time_steps_dqn = n_time_steps_dqn
        self.max_len_path = max_len_path

        self.gamma = gamma
        self.learning_rate_init = learning_rate_init

        self.epsilon_init = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.delta_init = delta_init
        self.delta_decay = delta_decay
        self.delta_min = delta_min
        
        self.max_norm=max_norm
        self.do_gradient_clipping = do_gradient_clipping

        self.tau = tau
        self.freq_update_target = freq_update_target
        self.targetNet_update_method = targetNet_update_method

        self.Train_Deep_Q_Learning = Train_Deep_Q_Learning
        self.saving_freq_DQN = saving_freq_DQN
        self.test_freq_DQN = test_freq_DQN
        # self.input_dims = input_dims
        
        
    def get_channels_parameters(self):
        if self.type_channel == "IID":
            return self.N_R, self.N_T, self.N_RIS, self.type_channel, self.mean_channel, self.std_channel
        if self.type_channel == "half-spaced ULAs":
            return self.N_R, self.N_T, self.N_RIS, self.type_channel, self.paths, self.alpha
    
    def get_noise_parameters(self):
        return self.mean_noise, self.std_noise
    
    def get_symbols_parameters(self):
        return self.type_modulation 
    
    def get_codebook_parameters(self):
        return self.size_codebooks, self.type_codebooks
    
    def set_N_RIS(self,N_RIS:int):
        self.N_RIS = N_RIS
    
    def set_SNR(self, snr:int):
        self.SNR = snr
        # print(f'SNR = {snr}')

    def set_noise(self, mean_noise:float, std_noise:float):
        self.mean_noise = mean_noise
        self.std_noise = std_noise
        # print(f'std noise = {std_noise}')
    
    def get_q_learning_parameters(self):
        return {
            
            "n_episodes": self.n_episodes,
            "n_channels_train": self.n_channels_train_QL,
            "n_time_steps": self.n_time_steps_ql,
            "len_path": self.len_path,

            "SNR" : self.SNR,
            "snr_values": self.snr_values,
            "size_codebooks" :self.size_codebooks,
            "initial_q_value": self.initial_q_value,

            "learning_rate_init": self.learning_rate_init,
            "learning_rate_decay": self.learning_rate_decay,
            "learning_rate_min": self.learning_rate_min,

            "gamma": self.gamma,

            "epsilon_init" : self.epsilon_init,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,

            "delta_init": self.delta_init,
            "delta_decay": self.delta_decay,
            "delta_min": self.delta_min,

            "len_window_action": self.len_window_action,
            "saving_freq": self.saving_freq_QL,
            "test_freq": self.test_freq_QL,

            "precision": self.precision,
            "blabla_other_states": self.blabla_other_states,
            "min_representatives_q_learning_train": self.min_representatives_q_learning_train,
            "max_samples_q_learning_train": self.max_samples_q_learning_train,
            "min_representatives_q_learning_test": self.min_representatives_q_learning_test,
            "max_samples_q_learning_test": self.max_samples_q_learning_test,
        }
    
    def get_dqn_parameters(self):
        return {
            "gamma": self.gamma,
            "learning_rate_init": self.learning_rate_init,
            "params_list": self.params_list,
            "batch_size": self.batch_size,
            "replay_buffer_memory_size": self.replay_buffer_memory_size,
            "loss_fct": self.loss_fct,
            "n_epochs": self.n_epochs,
            "n_time_steps": self.n_time_steps_dqn,
            "freq_update_target": self.freq_update_target,
            "tau": self.tau,
            "max_len_path": self.max_len_path,
            "epsilon_init": self.epsilon_init,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "delta_init": self.delta_init,
            "delta_decay": self.delta_decay,
            "delta_min": self.delta_min,
            "train_or_test": self.train_or_test,
            "targetNet_update_method": self.targetNet_update_method,
            "Train_Deep_Q_Learning": self.Train_Deep_Q_Learning,
            "saving_freq": self.saving_freq_DQN,
            "test_freq": self.test_freq_DQN,
            "n_channels_train": self.n_channels_train_DQN,
            "max_norm": self.max_norm,
            "do_gradient_clipping": self.do_gradient_clipping,
            "snr_values": self.snr_values,
            "SNR" : self.SNR,
            "size_codebooks" :self.size_codebooks,
            # "input_dims" : self.input_dims,
        }
    
    def save_to_file(self, filename, params_type='dqn'):
        # Save the parameters to a file, based on the type
        params = self.get_dqn_parameters() if params_type == 'dqn' else self.get_q_learning_parameters()
        with open(filename, 'w') as file:
            for key, value in params.items():
                file.write(f"{key}: {value}\n")
                