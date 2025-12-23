import numpy as np
from config.parameters import Parameters

class Signal:

    """
    Sends a symbol/symbols from a given constellation = type_modulation 
    """
    
    def __init__(self, parameters:Parameters):
        self.parameters = parameters
        
    def get(self) : 
        return self.symbols
    
    def set_random(self):
        type_modulation = self.parameters.get_symbols_parameters()
        N_T = (self.parameters.get_channels_parameters())[1]
        if type_modulation == "BPSK":
            # Generate random symbols {-1, 1}
            symbols = np.random.choice([-1,1], size= (N_T, 1))
        self.symbols = symbols
        
    def set_pilots(self):
        type_modulation = self.parameters.get_symbols_parameters()
        N_T = (self.parameters.get_channels_parameters())[1]
        if type_modulation == "BPSK":
            # Generate random symbols {-1, 1}
            symbols = np.array([1]*N_T)
        self.symbols = symbols