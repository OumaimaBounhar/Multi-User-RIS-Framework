import numpy as np
from typing import List
from typing import Tuple

from config.parameters import Parameters
from dataset.monteCarlo import Dataset_probability 

class Probability:
    """Updates a probability distribution (probability of being the best codeword):
    Input: 
    -prior probability ; 
    -list of the higher elements of the prior probability (likeliest elements) to decrease computational complexity ; 
    -new sample received ; 
    -law of the channel (p(Y|phi,h) for all h) that will be computed with Monte Carlo sampling on h
    -mean and standard deviation of the modeled gaussian noise
    Output:
    -posterior probability ; 
    -list of the higher elements of the posterior probability (likeliest elements) 
    """
    def __init__(self,
                parameters:Parameters,
                Law:Dataset_probability,
                params_modeled_noise:Tuple[float,float],
                params_evol_matrix:int = 1,
                ):
        self.parameters = parameters
        self.noise_mean,self.noise_std = params_modeled_noise
        self.Law = Law
        self.F = Law.get_Data()
        self.params_evol_matrix = params_evol_matrix
        self.matrix_evolution_channel = params_evol_matrix*np.eye(parameters.size_codebooks[0]) + (1-params_evol_matrix)/(parameters.size_codebooks[0]-1)*np.ones((parameters.size_codebooks[0],parameters.size_codebooks[0]))
        self.M = self.matrix_evolution_channel
    
    def update(self,prior:np.ndarray,ordered_list:np.ndarray,new_sample:List[Tuple[float,int]])->Tuple[np.ndarray,np.ndarray]:
        """ Computes p(phi|Y,psi) for a given Y"""
        posterior = np.zeros(len(prior))
        for i in range(0,len(prior)):
            index = ordered_list[i]
            likelihood:float = 0
            F = (self.F)[index][1]
            p = (self.F)[index][0]
            for montecarlo in range(0,len(p)):
                likelihood_montecarlo:float = 0
                min_likelihood = float("inf")
                for sample_index in range(0,len(new_sample)):
                    Y,phi_index = new_sample[sample_index]
                    likelihood_montecarlo = likelihood_montecarlo + (abs(Y-F[montecarlo][phi_index])**2) 
                likelihood_montecarlo = 1/(likelihood_montecarlo+1e-2) 
                likelihood = likelihood + likelihood_montecarlo ## To prevent dividing by zero
            posterior[index] = likelihood*prior[index]
        posterior = posterior + 1e-8 ## To prevent dividing by zero
        posterior = posterior/sum(posterior)
        new_ordered_list = np.argsort(posterior)[::-1] 
        return posterior,new_ordered_list
    
    def update_proba_channel(self,prior:np.ndarray)->Tuple[np.ndarray,np.ndarray]:
        """ Updates p(phi) because the channel changed """
        posterior = np.dot(self.M,prior)
        ordered_list = np.argsort(posterior)[::-1] 
        return posterior,ordered_list
    
    def set_noise(self,params_modeled_noise):
        self.params_modeled_noise = params_modeled_noise
    
