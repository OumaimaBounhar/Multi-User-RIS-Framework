import numpy as np
from config.parameters import Parameters
from envs.codebooks import Codebooks
from envs.signal import Signal
from envs.channel import Channel

class Feedback:
    """To compute the received feedback (scalar or vector) that will be used to infer the parametrization of the RIS"""
    def __init__(self, parameters:Parameters,channel:Channel,codebooks:Codebooks,signal:Signal):
        self.parameters = parameters
        self.channel = channel
        self.codebooks = codebooks
        self.signal = signal
        
    def transmit(self,index_codeword:int,codebook_used:int = 0):
        """Transmit a symbol and receives a vector at the antenna (No noise) for:
        a given configuration at the RIS ( we give the index of the codeword as an input)
        Two codebooks: 0 = pilots or 1 communication"""
        params_channel = self.parameters.get_channels_parameters()
        # Sends a signal
        self.signal.set_random()
        #self.signal.set_pilots()
        s = self.signal.get()
        # Goes through the channel with the effect of the RIS
        h_T,h_R,h_D = self.channel.get_channel()
        phi = self.codebooks.get_codeword(codebook_used,index_codeword)
        psi = np.diagflat(phi)
        if params_channel[2] == 0:
            # No RIS
            y = np.dot(h_D,s)
        if params_channel[2] != 0:
            # With RIS
            h_phi_h = np.dot(np.dot(h_R, psi), h_T)
            #print(h_phi_h)
            y = np.dot(h_phi_h, s) + np.dot(h_D, s)
        self.y = y
        #print(np.dot(np.conj(y),y.T).item())
        
    def get_y(self, gaussian_noise = True):
        if gaussian_noise:
            #Add the noise (complex Gaussian (i.i.d.))
            mean_noise, std_noise = self.parameters.get_noise_parameters()
            params_channel = self.parameters.get_channels_parameters()
            w = np.random.normal(mean_noise, std_noise/math.sqrt(2), size= (params_channel[0],params_channel[1])) + 1j * np.random.normal(mean_noise, std_noise/math.sqrt(2), size= (params_channel[0],params_channel[1]))
            # print(f'Std_noise = {std_noise}')
            return self.y + w
        else:
            return self.y
        
    def get_feedback(self,additive_noise_feedback = False,noise = True):
        # additive_noise_feedback = False: The feedback will be (y+w)*(y+w)H
        # additive_noise_feedback = True :The feedback will be yyH + gaussian
        feedback = 'RSE'
        if feedback == 'RSE':
            mean_noise, std_noise = self.parameters.get_noise_parameters()
            params = self.parameters.get_channels_parameters()
            N_R = params[0:3][0]
            if noise:
                if additive_noise_feedback:
                    y = self.get_y(gaussian_noise=False)
                    return np.dot(np.conj(y).T,y).item().real + np.random.normal(mean_noise, std_noise*N_R)
                
                else:
                    y = self.get_y(gaussian_noise=True)
                    return np.dot(np.conj(y).T,y).item().real
            else:
                y = self.get_y(gaussian_noise=False)
                return np.dot(np.conj(y).T,y).item().real
    
    #def get_SNR(self):
        #received_signal_power = np.sum(np.abs(h_phi_h + self.h_d)**2 * np.abs(s)**2 )
        #noise_power = np.sum(np.abs(w)**2)
        #return received_signal_power / noise_power