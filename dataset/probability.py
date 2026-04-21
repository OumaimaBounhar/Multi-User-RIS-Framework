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
    
        # Pre-stack F into contiguous ndarrays for vectorised inference.
        # Works identically for both math (monteCarlo) and Sionna datasets
        # since both expose the same get_Data() structure.
        self.build_F_matrix()

    # ------------------------------------------------------------------
    # Pre-computation (called once at init)
    # ------------------------------------------------------------------

    def build_F_matrix(self) -> None:
        """Pre-stack MC response arrays into padded contiguous ndarrays.

        Each codeword class may have a different number of MC samples
        (this always happens with the Sionna dataset, whose generation loop
        fills classes at unequal rates). We therefore:
        1. Find MC_max = max MC count across all codewords.
        2. Pad shorter classes with zeros in both F_matrix and p_matrix.
        3. Store a boolean mask _F_valid of shape (N_codewords, MC_max)
            that is True only for real (non-padded) entries.

        The mask is applied in update() so padded slots never contribute
        to the likelihood sum.

        Sets
        ----
        self._F_matrix : ndarray, shape (N_codewords, MC_max, N_pilots)
        self._p_matrix : ndarray, shape (N_codewords, MC_max)
        self._F_valid  : bool ndarray, shape (N_codewords, MC_max)
        """
        N      = len(self.F)
        MC_max = max(len(self.F[cw][0]) for cw in range(N))

        # Infer N_pilots from the first non-empty entry
        N_pilots = len(np.asarray(self.F[0][1][0]))

        F_matrix = np.zeros((N, MC_max, N_pilots), dtype=np.float32)
        F_valid  = np.zeros((N, MC_max),           dtype=bool)

        for cw in range(N):
            mc_count = len(self.F[cw][0])
            F_valid[cw, :mc_count] = True
            for mc in range(mc_count):
                F_matrix[cw, mc] = np.asarray(self.F[cw][1][mc], dtype=np.float32)

        self._F_matrix = F_matrix   # (N_codewords, MC_max, N_pilots)
        self._F_valid  = F_valid    # (N_codewords, MC_max) — False for padded slots

    # ------------------------------------------------------------------
    # Main inference — fully vectorised, no Python loops over MC or samples
    # ------------------------------------------------------------------

    def update(
        self,
        prior: np.ndarray,
        ordered_list: np.ndarray,
        new_sample: List[Tuple[float, int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes p(phi | Y, psi) for a given observation sequence.

        Vectorised implementation — replaces the original triple nested loop.
        Public signature is unchanged; fully drop-in compatible with all
        existing callers in env.py, methods.py, and test.py.

        Parameters
        ----------
        prior        : (N,) probability vector over codewords
        ordered_list : (N,) codeword indices sorted by descending prior
        new_sample   : list of (Y, phi_index) tuples accumulated since last reset

        Returns
        -------
        posterior        : (N,) updated and normalised probability vector
        new_ordered_list : (N,) indices sorted by descending posterior
        """
        if len(new_sample) == 0:
            return prior.copy(), np.argsort(prior)[::-1]

        samples_Y   = np.array([s[0] for s in new_sample], dtype=np.float32)  # (S,)
        samples_phi = np.array([s[1] for s in new_sample], dtype=np.int32)    # (S,)

        # F_sel: (N_codewords, MC_max, S) — pilot response for each observed action
        F_sel = self._F_matrix[:, :, samples_phi]

        # Squared residuals summed over all S observations → (N_codewords, MC_max)
        residuals = np.abs(samples_Y[np.newaxis, np.newaxis, :] - F_sel)
        sq_err    = np.sum(residuals ** 2, axis=2)

        # Inverse error per MC sample → (N_codewords, MC_max)
        # inv_err = self._p_matrix / (sq_err + 1e-2)
        inv_err = 1.0 / (sq_err + 1e-2)

        # Zero out padded slots so they don't contribute to the likelihood sum
        inv_err[~self._F_valid] = 0.0

        # Sum over MC → (N_codewords,)
        likelihood = np.sum(inv_err, axis=1)

        posterior  = likelihood * prior
        posterior += 1e-8           # prevent divide-by-zero (matches original)
        posterior /= posterior.sum()

        new_ordered_list = np.argsort(posterior)[::-1]
        return posterior, new_ordered_list

    # def update(
    #     self,
    #     prior:np.ndarray,
    #     ordered_list:np.ndarray,
    #     new_sample:List[Tuple[float,int]]
    # )->Tuple[np.ndarray,np.ndarray]:
    #     """
    #     Computes p(phi|Y,psi) for a given Y
    #     """
    #     posterior = np.zeros(len(prior))

    #     for i in range(0,len(prior)):

    #         index = ordered_list[i]

    #         likelihood:float = 0

    #         F = (self.F)[index][1]
    #         p = (self.F)[index][0]

    #         for montecarlo in range(0,len(p)):

    #             likelihood_montecarlo:float = 0

    #             min_likelihood = float("inf")

    #             for sample_index in range(0,len(new_sample)):

    #                 Y,phi_index = new_sample[sample_index]
    #                 likelihood_montecarlo = likelihood_montecarlo + (abs(Y-F[montecarlo][phi_index])**2) 

    #             likelihood_montecarlo = 1/(likelihood_montecarlo+1e-2) 
    #             likelihood = likelihood + likelihood_montecarlo ## To prevent dividing by zero

    #         posterior[index] = likelihood*prior[index]

    #     posterior = posterior + 1e-8 ## To prevent dividing by zero
    #     posterior = posterior/sum(posterior)

    #     new_ordered_list = np.argsort(posterior)[::-1] 

    #     return posterior,new_ordered_list
    
    # ------------------------------------------------------------------
    # Channel evolution update: update the matrix M that transforms the prior into the posterior to take into account the channel evolution.
    # ------------------------------------------------------------------

    def update_proba_channel(
        self,prior:np.ndarray
    )->Tuple[np.ndarray,np.ndarray]:
        """
        Updates p(phi) because the channel changed.
        """
        posterior = np.dot(self.M,prior)
        ordered_list = np.argsort(posterior)[::-1] 
        return posterior,ordered_list
    
    def set_noise(
        self,
        params_modeled_noise
    ):
        self.params_modeled_noise = params_modeled_noise
    
