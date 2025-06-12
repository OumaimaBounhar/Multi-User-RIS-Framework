import numpy as np
from config.parameters import Parameters

class Codebooks:
    """The codebooks that contains the matrices representing the coefficients of reflection of the RIS:
    First codebook: For the communication
    Second codebook: For the pilots
    """  
    def __init__(self, parameters:Parameters):
        self.parameters = parameters
        self.set_codebooks()
    
    def set_codebooks(self):
        size_codebooks, type_codebooks = self.parameters.get_codebook_parameters()
        
        N_RIS = (self.parameters.get_channels_parameters())[2]
        
        codebooks = []
        
        for nb_codebook in range(0,len(type_codebooks)):
            if type_codebooks[nb_codebook] == "random" :
                # Random Values for all reflective elements
                codebook = []
                for nb_codeword in range(size_codebooks[nb_codebook]):
                    #Create a random codeword of size (N_RIS,1)
                    codeword = np.random.uniform(low= 0, high= 2*np.pi, size= N_RIS)
                    codebook.append(codeword)

            if type_codebooks[nb_codebook] == "DFT" :
                # DFT Codebook
                #Generate a list of evenly spaced values of theta between 0:2*pi
                List_theta = (2*np.pi/size_codebooks[nb_codebook]) * np.arange(size_codebooks[nb_codebook])
                codebook = []
                for nb_codeword in range(0,size_codebooks[nb_codebook]):
                    #Create a codeword of size (N_RIS,1)
                    codeword = [np.exp(1j * index * List_theta[nb_codeword]) for index in range(0,N_RIS)]
                    codebook.append(codeword)
                    
            if type_codebooks[nb_codebook][:12] == "Hierarchical" :
                # Hierarchical Codebook, see def hierarchical_beam
                K,M = find_K_M_hierarchical(type_codebooks[nb_codebook])
                codebook = []
                for k in range(1,K+1):
                    for m in range(0,M**k):
                        codeword = self.hierarchical_beam(M,m,N_RIS,k)
                        codebook.append(codeword)
                        
            if type_codebooks[nb_codebook][:6] == "Narrow" :
                # Create a narrow beam, ( the last layer of a hierarchical codebook)
                N = find_N_narrow(type_codebooks[nb_codebook])
                codebook = []
                for n in range(0,N):
                    codeword = self.narrow_beam(n,N_RIS,N)
                    codebook.append(codeword)
                
            codebooks.append(codebook)
            
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

    def get_codeword(self, codebook_index:int,codeword_index:int) :
        """Return the codeword number:codebook_index ,of the codebook of communication (codeword_index=0) or pilots (codeword_index=1)"""
        return self.codebooks[codebook_index][codeword_index] 

def find_K_M_hierarchical(type_codebooks)->Tuple[int,int]:
    """From the name of the codebook simply read the parameters of the hierarchical codebook"""
    count = 0
    while type_codebooks[count+13].isdigit():
        count+=1
    K = int(type_codebooks[13:13+count])
    count2 = 1
    while type_codebooks[count+13+count2].isdigit():
        count2+=1
        if count+13+count2 == len(type_codebooks):
            break
    M = int(type_codebooks[13+count+1:13+count+count2+1])
    return K,M

def find_N_narrow(type_codebooks)->int:
    """From the name of the codebook simply read the parameters of the narrow codebook"""
    count = 0
    while type_codebooks[count+7].isdigit():
        count+=1
        if count+7 == len(type_codebooks):
            break
    N = int(type_codebooks[7:7+count])
    return N
    
def check_size_cd(type_codebooks,size_codebooks)->Tuple[List[int],bool]:
    """Check and correct the size of the codebooks,
    Also checks if a hierarchical search is possible with the given codebooks"""
    init_size_codebooks = size_codebooks
    for type in range(0,2):
        if type_codebooks[type][:12] == "Hierarchical":
            K,M = find_K_M_hierarchical(type_codebooks[type])
            size_codebooks[type] = sum([M**k for k in range(1,K+1)])
            
        if type_codebooks[type][:6] == "Narrow" :
            N = find_N_narrow(type_codebooks[type])
            size_codebooks[type] = N
            
    if type_codebooks[0][:6] == "Narrow" and type_codebooks[1][:12] == "Hierarchical":
        K,M = find_K_M_hierarchical(type_codebooks[1])
        N = find_N_narrow(type_codebooks[0])
        if N == M**K:
            Hierarchical_possible = True
        else:
            Hierarchical_possible = False
            print("Check the dimensions of your codebooks, Narrow_N and Hierarchical_k_M should respect N = M**K")
    else:
        Hierarchical_possible = False
        
    if not Hierarchical_possible:
        print("Hierarchical search not defined for those codebooks, look at the function check_size_cd")
    if size_codebooks != init_size_codebooks:
        print("size_codebooks was wrong, it was changed to:")
        #print(size_codebooks)
    return size_codebooks,Hierarchical_possible