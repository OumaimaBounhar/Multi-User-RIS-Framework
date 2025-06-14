from typing import List
from typing import Tuple
import math
import numpy as np


class Parameters :
    """
    Contains all of the parameters for our simulation
    """
    def __init__(   self, 
                    n_receivers: int = 64, 
                    n_transmitters: int = 1, 
                    n_RIS_elements: int = 100, 
                    size_codebooks: List[int] = [20,20], 
                    type_codebooks: List[str] = ["Narrow_8","Hierarchical_3_2"],
                    mean_noise: float = 0,
                    SNR: int = 30,
                    snr_values: List[int] = [-100,-50,0],
                    type_channel:str = "half-spaced ULAs",
                    type_modulation: str = "BPSK", 
                    mean_channel: float = 0, 
                    std_channel: List[float] = [], 
                    std_alpha = 0, 
                    gamma: float = 0.99,
                    learning_rate_init: float = 5e-4,
                    params_list: List[int] = [32, 64],
                    batch_size: int = 128,
                    replay_buffer_memory_size: int = None,
                    max_norm: float =1,
                    do_gradient_clipping: bool = True,
                    loss_fct: str = 'mse',
                    n_epochs: int = 10000,
                    n_time_steps_dqn: int = 200,
                    n_channels_train_DQN: int = 10,
                    freq_update_target: int = 10,
                    max_len_path: int = 20,
                    epsilon: float = 1,
                    epsilon_decay: float = 0.999,
                    epsilon_min: float = 0.01,
                    delta_init: float = 1e-1,
                    delta_decay: float = 1,
                    train_or_test: bool = True,
                    Train_Deep_Q_Learning: bool = True,
                    saving_freq_DQN: int = 1,
                    test_freq_DQN: int = 1,
                    Train_Q_Learning: bool = False,
                    n_episodes: int = 20,
                    n_time_steps_ql: int = 100,
                    n_channels_train_QL: int = 10,
                    len_path: int = 10,
                    saving_freq_QL: int = 1,
                    test_freq_QL: int = 1,
                    delta_final: float = 5 * 1e-2, 
                    precision: int = 2,  
                    min_representatives_q_learning_train: int = 100,  
                    min_representatives_q_learning_test: int = 10,
                    learning_rate_decay: float = 0.99,
                    learning_rate_min: float = 1e-4
                    ) :
        
        ### For the channel ###
        self.n_receivers = n_receivers # Number antennas at the Receiver
        self.n_transmitters = n_transmitters # Number antennas at the Transmitter
        self.n_RIS_elements = n_RIS_elements  # Number Reflective elements at the RIS
        self.std_alpha = std_alpha # Variance of the attenuation of paths 
        if type_channel == "IID":
            self.mean_channel = mean_channel
            self.std_channel = std_channel
        if type_channel == "half-spaced ULAs":
            self.paths = [1,1,0] # Number of paths for the link Transmitter-RIS, RIS-Receiver, Transmitter-Receiver
            self.alpha = np.array([[[1,0.3,0.3],[std_alpha]*3],[[1,0.3,0.3],[std_alpha]*3],[[0.5,0.3,0.3],[std_alpha]*3]]) # Attenuation of paths [1 for the LOS, less for NLOS]
        
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
        self.Train_Q_Learning = Train_Q_Learning
        self.n_episodes = n_episodes
        self.len_path = len_path
        self.n_channels_train_QL = n_channels_train_QL
        self.n_time_steps_ql = n_time_steps_ql
        self.initial_q_value = -len_path 
        self.saving_freq_QL= saving_freq_QL
        self.test_freq_QL = test_freq_QL
        self.delta_final = delta_final
        self.len_window_action = len_window_action
        self.precision = precision
        self.blabla_other_states = blabla_other_states
        self.min_representatives_q_learning_train = min_representatives_q_learning_train
        self.max_samples_q_learning_train = 2*min_representatives_q_learning_train*size_codebooks[0]
        self.min_representatives_q_learning_test = min_representatives_q_learning_test
        self.max_samples_q_learning_test = 2*min_representatives_q_learning_test*size_codebooks[0]
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_min = learning_rate_min
        
        ### For DQN Parameters ###
        self.gamma = gamma
        self.learning_rate_init = learning_rate_init
        self.params_list = params_list
        self.batch_size = batch_size
        self.replay_buffer_memory_size = replay_buffer_memory_size
        self.max_norm=max_norm
        self.do_gradient_clipping = do_gradient_clipping
        self.loss_fct = loss_fct
        self.n_epochs = n_epochs
        self.n_time_steps_dqn = n_time_steps_dqn
        self.n_channels_train_DQN = n_channels_train_DQN
        self.freq_update_target = freq_update_target
        self.max_len_path = max_len_path
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.delta_init = delta_init
        self.delta_decay = delta_decay
        self.train_or_test = train_or_test
        self.Train_Deep_Q_Learning = Train_Deep_Q_Learning
        self.saving_freq_DQN = saving_freq_DQN
        self.test_freq_DQN = test_freq_DQN

    def get_channels_parameters(self):
        if self.type_channel == "IID":
            return self.n_receivers, self.n_transmitters, self.n_RIS_elements, self.type_channel, self.mean_channel, self.std_channel
        if self.type_channel == "half-spaced ULAs":
            return self.n_receivers, self.n_transmitters, self.n_RIS_elements, self.type_channel, self.paths, self.alpha

    def get_noise_parameters(self):
        return self.mean_noise, self.std_noise

    def get_symbols_parameters(self):
        return self.type_modulation 

    def get_codebook_parameters(self):
        return self.size_codebooks, self.type_codebooks

    def set_n_RIS_elements(self,n_RIS_elements:int):
        self.n_RIS_elements = n_RIS_elements

    def set_SNR(self, snr:int):
        self.SNR = snr

    def set_noise(self, mean_noise:float, std_noise:float):
        self.mean_noise = mean_noise
        self.std_noise = std_noise

    def get_q_learning_parameters(self):
        return {
            "gamma": self.gamma,
            "n_episodes": self.n_episodes,
            "len_path": self.len_path,
            "n_time_steps": self.n_time_steps_ql,
            "initial_q_value": self.initial_q_value,
            "learning_rate_init": self.learning_rate_init,
            "epsilon" : self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "delta_init": self.delta_init,
            "delta_final": self.delta_final,
            "delta_decay": self.delta_decay,
            "len_window_action": self.len_window_action,
            "saving_freq": self.saving_freq_QL,
            "test_freq": self.test_freq_QL,
            "precision": self.precision,
            "blabla_other_states": self.blabla_other_states,
            "min_representatives_q_learning_train": self.min_representatives_q_learning_train,
            "max_samples_q_learning_train": self.max_samples_q_learning_train,
            "min_representatives_q_learning_test": self.min_representatives_q_learning_test,
            "max_samples_q_learning_test": self.max_samples_q_learning_test,
            "n_channels_train": self.n_channels_train_QL,
            "learning_rate_decay": self.learning_rate_decay,
            "learning_rate_min": self.learning_rate_min,
            "snr_values": self.snr_values,
            "SNR" : self.SNR,
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
            "max_len_path": self.max_len_path,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "delta_init": self.delta_init,
            "delta_decay": self.delta_decay,
            "train_or_test": self.train_or_test,
            "Train_Deep_Q_Learning": self.Train_Deep_Q_Learning,
            "saving_freq": self.saving_freq_DQN,
            "test_freq": self.test_freq_DQN,
            "n_channels_train": self.n_channels_train_DQN,
            "max_norm": self.max_norm,
            "do_gradient_clipping": self.do_gradient_clipping,
            "snr_values": self.snr_values,
            "SNR" : self.SNR,
        }

    def save_to_file(self, filename, params_type='dqn'):
        # Save the parameters to a file, based on the type
        params = self.get_dqn_parameters() if params_type == 'dqn' else self.get_q_learning_parameters()
        with open(filename, 'w') as file:
            for key, value in params.items():
                file.write(f"{key}: {value}\n")
                