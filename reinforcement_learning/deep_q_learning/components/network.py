import torch.nn as nn
from config.parameters import Parameters

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
        params_list = params_dict["params_list"]
        
        #Layers of Neural Network
        layers = []
        
        prev_dim = input_dims
        for hidden_dim in params_list:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, n_actions))
        
        self.network = nn.Sequential(*layers)
        
        print(f'DQN network: {self.network}')
        
        #Loss function
        if loss_fct == 'mse':
            self.loss_fct = nn.MSELoss()
        
    def forward(self, state):
        q_value = self.network(state)
        return q_value
