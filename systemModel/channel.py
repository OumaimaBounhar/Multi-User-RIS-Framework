import numpy as np
from config.parameters import Parameters

class Channel:
    """Create the channel with:
    N_T = number of antennas of the Transmitter, N_R = Number of antennas at the receiver, N_RIS = Number of reflective elements on the RIS,
    type_channel is "IID" => with parameters mean_channel and std_channel /or "half-spaced ULAs" => with parameters paths and lambda
    """
    def __init__(self, parameters:Parameters):    
        self.parameters = parameters
    
    def set_channel(self):
        """Sets the channel matrices to some value depending on the channel model:
        half-spaced ULAs: h_1_2 = sum_l(alpha_l*a^H(phi1_l)*a(phi2_l))"""
        params = self.parameters.get_channels_parameters()
        N_R, N_T, N_RIS = params[0:3]
        if params[3] == "half-spaced ULAs":
            Angles = self.Angles
            Attenuation = self.Attenuation
            
            h_R = np.zeros((N_R,N_RIS),dtype="complex")
            h_T = np.zeros((N_RIS,N_T),dtype="complex")
            
            if N_RIS !=0:
                for i in range(params[4][0]):
                    #a_1 = np.array([[np.exp(1j * np.pi * index * np.sin(Angles[0][0][i])) for index in range(0,N_R)]])/math.sqrt(N_R)
                    a_1 = np.array([[np.exp(1j * np.pi * index * np.sin(Angles[0][0][i])) for index in range(0,N_R)]])
                    #a_2 = np.array([[np.exp(1j * np.pi * index * np.sin(Angles[0][1][i])) for index in range(0,N_RIS)]])/math.sqrt(N_RIS)
                    a_2 = np.array([[np.exp(1j * np.pi * index * np.sin(Angles[0][1][i])) for index in range(0,N_RIS)]])
                    att = Attenuation[0][i]
                    #att = 100
                    h_R += att * np.dot(np.conj(a_1).T,np.array(a_2))
                for i in range(params[4][1]):
                    #a_1 = np.array([[np.exp(1j * np.pi * index * np.sin(Angles[1][0][i])) for index in range(0,N_RIS)]])/math.sqrt(N_RIS)
                    a_1 = np.array([[np.exp(1j * np.pi * index * np.sin(Angles[1][0][i])) for index in range(0,N_RIS)]])
                    #a_2 = np.array([[np.exp(1j * np.pi * index * np.sin(Angles[1][1][i])) for index in range(0,N_T)]])/math.sqrt(N_T)
                    a_2 = np.array([[np.exp(1j * np.pi * index * np.sin(Angles[1][1][i])) for index in range(0,N_T)]])
                    att = Attenuation[1][i]
                    #att = 100
                    h_T += att * np.dot(np.conj(a_1).T,np.array(a_2))
            else:
                h_R = 0
                h_T = 0
            h_D = np.zeros((N_R,N_T),dtype="complex")
            for i in range(params[4][2]):
                #a_1 = np.array([[np.exp(1j * np.pi * index * np.sin(Angles[2][0][i])) for index in range(0,N_R)]])/math.sqrt(N_R)
                a_1 = np.array([[np.exp(1j * np.pi * index * np.sin(Angles[2][0][i])) for index in range(0,N_R)]])
                #a_2 = np.array([[np.exp(1j * np.pi * index * np.sin(Angles[2][1][i])) for index in range(0,N_T)]])/math.sqrt(N_T)
                a_2 = np.array([[np.exp(1j * np.pi * index * np.sin(Angles[2][1][i])) for index in range(0,N_T)]])
                att = Attenuation[2][i]
                #att = 100
                h_D += att * np.dot(np.conj(a_1).T,np.array(a_2))
            
        self.h_R = h_R
        self.h_T = h_T
        self.h_D = h_D
        
    def new_channel(self):
        """Creates a new channel randomly"""
        self.set_random_angles() ## New AoA/AoD
        self.set_random_attenuation() ## New attenuations
        self.set_channel()
        
    def update(self,modification_channel:float = 0):
        """Creates a new channel function of the previous one"""
        self.set_angles(modification_channel) ## Modifies AoA/AoD
        self.set_attenuation(modification_channel) ## Modifies attenuations
        self.set_channel()
        
    def set_random_angles(self):
        """This function sets the angles AoA/AoD for the channel half-spaced ULAs: Randomly"""
        angles = []
        params = self.parameters.get_channels_parameters()
        for channel in range(0,3): # 3 because Transmit Received and Direct
            angles.append([np.random.uniform(0,2*np.pi,params[4][channel]), np.random.uniform(0,2*np.pi,params[4][channel])])
            #angles.append([np.random.randint(9,size =params[4][channel])/2/math.pi,np.random.randint(9,size=params[4][channel])/2/math.pi])
        self.Angles = angles
        
    def set_random_attenuation(self):
        """This function sets the attenuation for the channel half-spaced ULAs: Randomly
        matrix values are:
        strength_path * normal(mean 1, variance sigma)"""
        att = []
        params = self.parameters.get_channels_parameters()
        for channel in range(0,3): # 3 because Transmit Received and Direct
            att.append(([np.random.normal(1, params[5][channel][1][i]) + 1j * np.random.normal(1, params[5][channel][1][i]) for i in range(0,len(params[5][channel][1]))])*params[5][channel][0]/math.sqrt(2))
        
        #print(params[5][channel])
        #print(att)
        self.Attenuation = att
        
    def set_angles(self,modification_channel):
        """This function sets the angles AoA/AoD for the channel half-spaced ULAs: Function of the previous ones"""
        params = self.parameters.get_channels_parameters()
        angles = self.Angles
        for channel in range(0,3): # 3 because Transmit Received and Direct
            angles[channel] = angles[channel] + [np.random.uniform(-modification_channel,modification_channel,params[4][channel]), np.random.uniform(-modification_channel,modification_channel,params[4][channel])]  
        self.Angles = angles
        
    def set_attenuation(self,modification_channel):
        """This function sets the attenuations for the channel half-spaced ULAs: Function of the previous ones"""
        params = self.parameters.get_channels_parameters()
        attenuation = self.Attenuation
        for channel in range(0,3): # 3 because Transmit Received and Direct
            attenuation[channel] = attenuation[channel] + np.random.uniform(-modification_channel,modification_channel,len(attenuation[channel])) + 1j * np.random.uniform(-modification_channel,modification_channel,len(attenuation[channel])) 
        self.Attenuation = attenuation    

    def get_channel(self):
        return self.h_T, self.h_R, self.h_D
    
    def get_angles(self):
        return self.Angles
