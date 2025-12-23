import numpy as np
import pickle
import pandas as pd
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
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
                    #likelihood_montecarlo = likelihood_montecarlo + abs(Y-F[montecarlo][phi_index]-self.noise_mean)**2/self.noise_std**2
                    likelihood_montecarlo = likelihood_montecarlo + (abs(Y-F[montecarlo][phi_index])**2) ### Before
                    #likelihood_montecarlo = likelihood_montecarlo + abs(Y-F[montecarlo][phi_index])**2 ## New
                    #likelihood_montecarlo = likelihood_montecarlo + abs(Y-F[montecarlo][phi_index])/Y
                    #likelihood_montecarlo = likelihood_montecarlo + abs(Y-F[montecarlo][phi_index])**2
                #likelihood_montecarlo = np.exp(-likelihood_montecarlo)
                #if likelihood_montecarlo < min_likelihood:
                    #min_likelihood = likelihood_montecarlo
                likelihood_montecarlo = 1/(likelihood_montecarlo+1e-2) ## Before To prevent dividing by zero
                likelihood = likelihood + likelihood_montecarlo ### Before###
            #likelihood = 1/(min_likelihood+1e-4) ### new ###
            #print(index)
            #print(likelihood)
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
        
"----------------------------------------------------------------------------------------------------------------------------------------------------------------------------"

def load_fitted_noise(filename:str):
    file = filename + "/Noise_parameters.csv"
    csv_noise = pd.read_csv(file)
    Noise_parameters = csv_noise.to_numpy()[:,1:]
    return Noise_parameters[0].item(),Noise_parameters[1].item()

# def fit_noise(filename:str,feedback:Feedback,channel:Channel,parameters:Parameters,max_samples=1000):
#     data = []
#     size_codebooks, type_codebooks = parameters.get_codebook_parameters()
#     print("Fitting Real noise to gaussian noise")
#     for sample in tqdm(range(max_samples)):
#         channel.new_channel()
#         for n_communications in range(0,size_codebooks[1]):
#             feedback.transmit(n_communications,codebook_used=1)
#             RSE = feedback.get_feedback(noise=True)
#             RSE_no_noise = feedback.get_feedback(noise=False)
#             data.append(RSE-RSE_no_noise)
#     res = scipy.stats.norm.fit(data)
#     res_pd = pd.DataFrame(np.array([res[0],res[1]]))
#     res_pd.to_csv(filename + "/Noise_parameters.csv")
#     return res

# def fit_noise(filename:str,feedback:Feedback,channel:Channel,parameters:Parameters,max_samples=1000):
#     data = []
#     Noise_Contribution = []
#     size_codebooks, _ = parameters.get_codebook_parameters()
#     print("Fitting Real noise to gaussian noise")
    
#     for sample in tqdm(range(max_samples)):
#         channel.new_channel()
#         for n_communications in range(size_codebooks[1]):
#             feedback.transmit(n_communications, codebook_used=1)
#             RSE = feedback.get_feedback(noise=True)
#             RSE_no_noise = feedback.get_feedback(noise=False)
#             data.append(RSE - RSE_no_noise)
#             Noise_Contribution.append((RSE - RSE_no_noise)/RSE)
            
#     res = scipy.stats.norm.fit(data)
#     res_pd = pd.DataFrame(np.array([res[0],res[1]]))
#     res_pd.to_csv(filename + "/Noise_parameters.csv")
    
#     # Plot histogram
#     plt.figure(figsize=(10, 6))
#     plt.hist(Noise_Contribution, bins=30, density=True, alpha=0.6, color='b', edgecolor='black', label='Noise Contribution')
    
#     # Plot fitted Gaussian curve
#     xmin, xmax = plt.xlim()
#     x = np.linspace(xmin, xmax, 100)
#     p = scipy.stats.norm.pdf(x, res[0], res[1])
#     plt.plot(x, p, 'r', linewidth=2, label='Fitted Gaussian')
    
#     plt.title('Histogram of Noise Contribution with Gaussian Fit')
#     plt.xlabel('Noise Contribution')
#     plt.ylabel('Density')
#     plt.legend()
#     plt.grid(True)
    
#     plt.savefig(filename + "/Noise_histogram.png")
#     plt.close()
    
#     return res

import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from tqdm import tqdm

def fit_noise(filename: str, feedback, channel, parameters, max_samples=1000):
    print("Fitting real noise to Gaussian noise")

    size_codebooks, _ = parameters.get_codebook_parameters()
    noise_results = []

    for snr in parameters.snr_values:
        print(f"\nProcessing SNR = {snr} dB...")
        
        # Set the SNR for the current iteration
        parameters.set_SNR(snr)
        parameters.std_noise = np.sqrt(10 ** (-snr / 10))
        parameters.set_noise(mean_noise=0, std_noise=parameters.std_noise)

        data = []
        noise_contribution = []

        for sample in tqdm(range(max_samples), desc=f"SNR {snr} dB"):
            channel.new_channel()
            for n_communications in range(size_codebooks[1]):
                feedback.transmit(n_communications, codebook_used=1)
                RSE = feedback.get_feedback(noise=True)
                RSE_no_noise = feedback.get_feedback(noise=False)

                noise_value = RSE - RSE_no_noise
                data.append(noise_value)
                noise_contribution.append(np.log(RSE_no_noise / noise_value if noise_value != 0 else 0))

        # Fit noise to a Gaussian distribution
        mu, sigma = scipy.stats.norm.fit(data)
        noise_results.append([snr, mu, sigma])

        # Save parameters for this SNR
        res_df = pd.DataFrame([[snr, mu, sigma]], columns=["SNR (dB)", "Mean", "Std"])
        res_df.to_csv(f"{filename}/Noise_parameters_snr_{snr}.csv", index=False)

        # Plot histogram of noise contribution
        plt.figure(figsize=(10, 6))
        plt.hist(noise_contribution, bins=50, density=True, alpha=0.6, color='b', edgecolor='black', label='Noise Contribution')
        plt.title(f'Histogram of Noise Contribution (SNR={snr} dB)')
        plt.xlabel('Noise Contribution')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)

        # Save histogram for this SNR
        plt.savefig(f"{filename}/Noise_histogram_snr_{snr}.png")
        plt.close()

    # Save all noise fitting results in one file
    noise_results_df = pd.DataFrame(noise_results, columns=["SNR (dB)", "Mean", "Std"])
    noise_results_df.to_csv(f"{filename}/Noise_parameters_all.csv", index=False)

    return noise_results_df
