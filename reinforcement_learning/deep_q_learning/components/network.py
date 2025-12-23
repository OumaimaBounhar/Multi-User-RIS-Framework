import numpy as np 
from tqdm import tqdm
import torch
import os
import torch.nn as nn
from reinforcement_learning.env import Environment 
from config.parameters import Parameters
from matplotlib import pyplot as plt 

class DQN(nn.Module):
    """ Class for Deep Q-Learning Network"""
    
    def __init__(   self, 
                    parameters: Parameters,
                    input_dims : int,
                    n_actions : int
                ):
        
        super(DQN, self).__init__()
        
        self.simu_parameters = parameters
        params_dict = self.simu_parameters.get_dqn_parameters()
        
        loss_fct = params_dict["loss_fct"]
        learning_rate = params_dict["learning_rate_init"]
        params_list = params_dict["params_list"]
        
        self.device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
        
        #Layers of Neural Network
        layers = []
        
        for i in range(len(params_list)-1):
            
            if i == 0:
                layers.append(nn.Linear(input_dims,  params_list[i]))
            else:
                layers.append(nn.Linear(params_list[i-1],  params_list[i]))
            layers.append(nn.ReLU())
            
        # Output layer
        layers.append(nn.Linear(params_list[-1], n_actions))
        
        self.network = nn.Sequential(*layers).to(self.device)
        
        print(f'DQN network: {self.network}')
        
        #Loss function
        if loss_fct == 'mse':
            self.loss_fct = nn.MSELoss().to(self.device)
            
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = learning_rate, weight_decay=1e-4)
        
    def forward(self, state):
        q_value = self.network(state)
        return q_value