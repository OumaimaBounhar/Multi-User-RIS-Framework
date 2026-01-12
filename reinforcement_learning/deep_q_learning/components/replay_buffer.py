import numpy as np 
from tqdm import tqdm
import torch
import os
import torch.nn as nn
from reinforcement_learning.env import Environment 
from config.parameters import Parameters
from matplotlib import pyplot as plt 

class ReplayBuffer():
    """Class for Experience Replay Buffer"""
    
    def __init__(self, replay_buffer_memory_size: int, batch_size: int, input_dims, device = 'cpu'):
        self.replay_buffer_memory_size = replay_buffer_memory_size
        self.batch_size = batch_size
        self.memory_counter = 0
        self.input_dims = input_dims
        
        #Initialize the replay buffer tensors
        self.current_state_memory = np.zeros([replay_buffer_memory_size, input_dims], dtype=np.float32) # Make the buffer float32 (saves ~50% on memory)
        self.next_state_memory = np.zeros([replay_buffer_memory_size, input_dims], dtype=np.float32)
        self.action_memory = np.zeros([replay_buffer_memory_size,1], dtype=np.int64)
        self.reward_memory = np.zeros([replay_buffer_memory_size,1], dtype=np.float32)
        
        #Index to keep track of where to store the data
        self.idx = 0
        
    def store_transition(self, current_state, action, reward, next_state):
        # Update the index, wrapping around when it exceeds the memory size
        self.idx = self.memory_counter % self.replay_buffer_memory_size
        
        #Store the transitions
        self.current_state_memory[self.idx] = current_state
        self.action_memory[self.idx] = action
        self.reward_memory[self.idx] = reward
        self.next_state_memory[self.idx] = next_state
        
        #Increment the counter
        self.memory_counter += 1
        
    def sample_buffer(self):
        max_value = min(self.memory_counter, self.replay_buffer_memory_size)
        
        batch_indices = np.random.choice(max_value, self.batch_size, replace= False)
        
        current_state_batch = self.current_state_memory[batch_indices]
        next_state_batch = self.next_state_memory[batch_indices]
        action_batch = self.action_memory[batch_indices]
        reward_batch = self.reward_memory[batch_indices]
        
        #self.reinitialize_buffer()
        return current_state_batch, action_batch, reward_batch, next_state_batch