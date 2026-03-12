import math
import numpy as np
from dataclasses import dataclass
from config.parameters import Parameters
from typing import Optional, List, Tuple, Callable, Dict

@dataclass
class CodebookSpec:
    kind: str                         # "random" | "dft" | "hierarchical" | "narrow"
    N: Optional[int] = None           # for random/dft/narrow
    K: Optional[int] = None           # hierarchical
    M: Optional[int] = None           # hierarchical

class Codebooks:
    """The codebooks that contains the matrices representing the coefficients of reflection of the RIS:
    First codebook: For the communication
    Second codebook: For the pilots
    """  
    def __init__(self, parameters:Parameters, seed: int):
        self.parameters = parameters
        self.rng = np.random.default_rng(seed)
        self.set_codebooks()
    
    def set_codebooks(self):
        size_codebooks, codebook_specs = self.parameters.get_codebook_parameters()
        N_RIS = self.parameters.get_channels_parameters()[2]
        
        codebooks = []
        
        for size, spec in zip(size_codebooks, codebook_specs):
            kind = spec.kind.lower()
            
            if kind == "random":
                # Generate random values for all reflective elements
                codebook = []
                #Create a random codeword of size (N_RIS,1)
                angles = self.rng.uniform(0, 2*np.pi, size=(size, N_RIS))
                codebook.append(np.exp(1j * angles))  # store complex

            elif kind == "dft":
                # Generate a DFT Codebook
                #Generate a list of evenly spaced values of theta between 0:2*pi
                theta = (2*np.pi/size) * np.arange(size)
                codebook = []
                for nb_codeword in range(size):
                    codeword = np.exp(1j * np.arange(N_RIS) * theta[nb_codeword])
                    codebook.append(codeword)

            elif kind == "hierarchical":
                # Generate a hierarchical codebook, cf. def hierarchical_beam
                K, M = spec.K, spec.M
                if K is None or M is None:
                    raise ValueError("hierarchical spec needs K and M")
                codebook = []
                for k in range(1, K + 1):
                    for m in range(M**k):
                        codebook.append(self.hierarchical_beam(M, m, N_RIS, k))
            
            elif kind == "narrow":
                N = spec.N
                if N is None:
                    raise ValueError("narrow spec needs N")
                codebook = []
                for n in range(N):
                    codebook.append(self.narrow_beam(n, N_RIS, N))
            
            else:
                raise ValueError(f"Unknown codebook kind: {spec.kind}")
            
            if len(codebook) != size:
                raise ValueError(f"Size mismatch: asked size={size}, generated={len(codebook)} for {spec}")
        
            codebooks.append(codebook)

            # debug: show lengths of generated codebooks vs expected sizes
            print([len(cb) for cb in codebooks], "expected", size_codebooks)
            
        self.codebooks = codebooks
        
    def hierarchical_beam(self,M,number,N,k):
        """Create a beam from a hierarchical codebook with phase deactivation
        Similar to: B.Ning,Terahertz Multi User Massive MIMO.../ Z.Xiao Hierarchical Codebook Design for Beamforming....
        M is tree type, example: 2(binary tree), 3
        k: current stage, which layer of the tree
        number : which beam at stage k: from 0 to M**k
        N: size of the vector
        """
        I = [-1+2/M*index+1/M for index in range(0,M)]
        angle = 0
        for k_previous in range(k, 0, -1):
            angle += I[number%M]/M**(k_previous-1) 
            number = number//M
        output = [np.exp(1j*math.pi*angle*n) for n in range(0,min(int(M**(k)),N))]
        output += [0]*max(N-int(M**(k)),0)
        return np.array(output)
    
    def narrow_beam(self,N_index,N_RIS,N_beams):
        """Create a narrow beam from the hierarchical codebook defined in hierarchical_beam
        N_beams is the number of elements in the codebook
        N_index is the index of the codeword chosen
        """
        I = [-1+2/N_beams*index+1/N_beams for index in range(0,N_beams)]
        angle = I[N_index]
        output = [np.exp(1j*math.pi*angle*n) for n in range(0,min(N_beams,N_RIS))]
        output += [0]*max(N_RIS-N_beams,0)
        return np.array(output)

    def get_codebooks(self)->list:
        """Return the codebooks"""
        return self.codebooks

    def get_codeword(self, codebook_index:int, codeword_index:int) :
        """Return the codeword number:codebook_index ,of the codebook of communication (codeword_index=0) or pilots (codeword_index=1)"""
        return self.codebooks[codebook_index][codeword_index] 
