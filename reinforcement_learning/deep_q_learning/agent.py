import numpy as np 
from tqdm import tqdm
import torch
import os
import torch.nn as nn
from reinforcement_learning.env import Environment 
from config.parameters import Parameters
from matplotlib import pyplot as plt 

class DeepQLearningAgent():
    """ Class for Dep Q-Learning Algorithm"""
    
    def __init__(self,
                    input_dims: int, #we input the state index
                    environment : Environment,
                    parameters: Parameters,
                    name_file:str="Data/Example_0/DQN"
                ):
        
        self.simu_parameters = parameters
        self.name = name_file
        self.environment = environment
        self.input_dims = self.environment.get_size_states()
        self.n_states = self.environment.state_space.get_n_states()
        self.n_actions = self.environment.get_size_codebook()[1]
        
        params_dict = self.simu_parameters.get_dqn_parameters()
        # self.input_dims = params_dict["input_dims"]
        # print(f"input dims : {input_dims}")
        
        # n_states = params_dict["n_states"]
        # n_actions = params_dict["n_actions"]
        
        ## Parameters
        
        batch_size = params_dict["batch_size"]
        replay_buffer_memory_size = params_dict["replay_buffer_memory_size"]
        # Set the maximum allowed gradient norm
        
        self.delta_final = environment.get_delta_final() ## Final degree of precision we want to reach, should correspond to the one in states
    
        ## Dataset
        self.dataset_train,self.dataset_test = environment.get_dataset()
        
        self.evaluation_q_network = DQN(self.simu_parameters, self.input_dims, self.n_actions)
        self.target_q_network = DQN(self.simu_parameters, self.input_dims, self.n_actions)

        # the target network is initialized with the same weights as the evaluation network
        self.target_q_network.load_state_dict(self.evaluation_q_network.state_dict())
                
        self.replay_buffer = ReplayBuffer(replay_buffer_memory_size, batch_size, input_dims)
        
        self.policy = np.zeros([self.n_states, self.n_actions])
        
    def choose_action(self, state, epsilon):
        """ Choose an action following the Epsilon Greedy Policy"""
        if np.random.random() < epsilon :
            action = np.random.randint(self.n_actions)
        else :
            state_tensor = torch.tensor(state, dtype= torch.float32).to(self.evaluation_q_network.device)
            with torch.no_grad():
                q_values = self.evaluation_q_network(state_tensor)
                action = torch.argmax(q_values).item()
        return action
    
    def train(self, params_dict = {}, testing_objects_dict = {}):
        """" Train the Deep Q-Learning Network """
        
        batch_size = params_dict["batch_size"]
        epsilon_min = params_dict["epsilon_min"]
        n_epochs = params_dict["n_epochs"]
        n_time_steps = params_dict["n_time_steps"]
        epsilon_decay = params_dict["epsilon_decay"]
        max_len_path = params_dict["max_len_path"]
        train_or_test = params_dict["train_or_test"]
        saving_freq = params_dict["saving_freq"]
        test_freq = params_dict["test_freq"]
        epsilon = params_dict["epsilon"]
        n_channels_train = params_dict["n_channels_train"]
        freq_update_target = params_dict["freq_update_target"]
        tau = params_dict["tau"]
        
        print(f"Initialize Training for DQN Network on Device : {self.target_q_network.device}")
        
        epsilons = []
        avg_losses = []
        avg_len_path = []
        for epoch in tqdm(range(n_epochs)):
            
            all_len_path = []
            losses = []
            
            for channel_realization in range(n_channels_train):
                
                current_state = self.environment.state_space.get_state_from_index(0) ## We always start at the Initial State
                length_path = 0 ## The length of the path we take to be able to measure how efficient the policy is
                
                ## Set the prior at the initial state
                self.environment.reset_prior()
                len_window_channel = self.environment.get_len_window_channel()
                
                ## Generates a new channel (take it randomly from the dataset given in the environment)
                if train_or_test:
                    index_class_channel = np.random.randint(0,len(self.dataset_train)) # Random class
                    index_specific_channel = np.random.randint(0,len((self.dataset_train)[index_class_channel][1])) # Random channel from this class
                else:
                    ## Test elements in the dataset one after the other
                    index_class_channel = time_step%len(self.dataset_test)
                    index_specific_channel = time_step//len(self.dataset_test)
                    if time_step//len(self.dataset_test) >= len((self.dataset_test)[index_class_channel][1]):
                        index_specific_channel = np.random.randint(0,len((self.dataset_test)[index_class_channel][1]))
                index_channel = (index_class_channel,index_specific_channel)
                
                for time_step in range(n_time_steps):
                    
                    for path in range(max_len_path):
                        # Choose an action following Epsilon greedy Policy
                        #current_state = self.environment.state_space.get_state_from_index(current_state_index)
                        #assert current_state.shape == (self.input_dims ,), f"Current state shape is incorrect: expected ({self.input_dims },), got {current_state.shape}"
                        if train_or_test:
                            index_action = self.choose_action(current_state, epsilon)
                        else:
                            index_action = self.choose_action(current_state, 0) ## Greedy Action during the test
                        
                        # Update the state
                        ## reset the list of samples and put prior = posterior
                        if path % len_window_channel == 0:
                            self.environment.reset_curse_dimension()
                            
                        next_state, reward = self.environment.step(index_channel,index_action,train_or_test=train_or_test, model_type = 'DQN')
                        
                        # Save the reward
                        if train_or_test:
                            length_path += reward
                        else:
                            length_path += -1
                            
                        # # Store the transition in the Replay Buffer if we train
                        if train_or_test:
                            self.replay_buffer.store_transition(current_state, index_action, reward, next_state)
                            
                        current_state = next_state
                        
                        # Stop condition in a terminal state
                        delta = self.environment.get_delta_current()
                        
                        if train_or_test:
                            #print(current_state)
                            if max(current_state)>=1-delta:
                                break 
                        else:
                            probability = self.environment.get_posterior()
                            if max(probability)>=1-delta:
                                break
                            
                    all_len_path.append(1-length_path)
                    
                    #print(self.replay_buffer.memory_counter)
                    if self.replay_buffer.memory_counter >= batch_size:
                        # Sample a batch from the Replay Buffer
                        current_state_batch, action_batch, reward_batch, next_state_batch = self.replay_buffer.sample_buffer()
                        
                        # Start the training process
                        loss_value = self.learn(current_state_batch, action_batch, reward_batch, next_state_batch,  params_dict = params_dict)
                        losses.append(loss_value)
                        
            # save the model checkpoints
            if epoch % saving_freq == 0:
                checkpoint_dir =  self.name + '/checkpoints/'
                os.makedirs(checkpoint_dir, exist_ok=True)
                model_path = checkpoint_dir + 'epoch_' + str(epoch) 
                torch.save(self.evaluation_q_network.state_dict(), model_path + '_eval.pth')
                torch.save(self.target_q_network.state_dict(), model_path + '_target.pth')
                print(f'Weights saved in: {model_path}')
            
            # if epoch % test_freq == 0:
            #     # run the test
            #     self.evaluation_q_network.eval()
            #     self.test(testing_objects_dict, epoch)
            #     self.evaluation_q_network.train()
            
            avg_losses.append(np.mean(losses) if len(losses) > 0 else np.nan)
            avg_len_path.append(np.mean(all_len_path))
            
            tqdm.write(f"Epoch {epoch}: Loss = {np.mean(losses):.6f}, Avg Path = {np.mean(all_len_path):.2f}, Epsilon = {epsilon:.3f}")

            # update the epsilon value
            epsilons.append(epsilon)
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            
            # update delta
            self.environment.delta = max(self.environment.delta*self.environment.delta_decay, self.delta_final)
            print(f'[INFO] Delta value : f{self.environment.delta}')
            
            # Update target network every `target_update_freq` epochs
            if epoch % freq_update_target == 0:
                # self.target_q_network.load_state_dict(self.evaluation_q_network.state_dict())
                self.soft_update(self.evaluation_q_network, self.target_q_network, tau) # tau = 1 if not using a soft update

        # print(f'[INFO] Best Score: {max(score_history)}')
        
        torch.save(self.evaluation_q_network.state_dict(), checkpoint_dir + 'last_eval.pth')
        torch.save(self.target_q_network.state_dict(), checkpoint_dir + 'last_target.pth')
        print("[INFO] Deep Q-Learning Training : Process Completed !")
                
        #mean_length_path = np.mean(all_len_path)
        
        # Extract Policy
        #states = self.environment.state_space.initialize()
        #policy = torch.zeros((self.n_states, self.n_actions), dtype= torch.int64).to(self.evaluation_q_network.device)
        
        #for idx, state in enumerate(states):
            #q_values = self.evaluation_q_network(state)
            #action = torch.argmax(q_values, dim = 1).item()
            #policy[idx, action] = 1  # Set the corresponding action to 1, indicating it's chosen by the policy
            
        #print("Extracted Policy:\n", policy)

        # Save loss values in a CSV file
        loss_file_path = os.path.join(self.name, "losses.csv")
        np.savetxt(loss_file_path, avg_losses, delimiter=",", header="Average Loss", comments="")
        
        # Save average len path values in a CSV file
        avg_len_path_file_path = os.path.join(self.name, "avgLenPath.csv")
        np.savetxt(avg_len_path_file_path, avg_len_path, delimiter=",", header="Average Len Path", comments="")

        # Save exploration rates in a CSV file
        epsilon_file_path = os.path.join(self.name, "epsilons.csv")
        np.savetxt(epsilon_file_path, epsilons, delimiter=",", header="Epsilon Values", comments="")

        print(f"[INFO] Losses saved to {loss_file_path}")
        print(f"[INFO] Epsilon values saved to {epsilon_file_path}")

        # Plot the convergence of the algorithm
        
        plt.plot(avg_losses, 'b', label='loss')
        #plt.plot(epsilons, 'r', label='Exploration Rate')
        plt.title("Convergence of DQN Algorithm loss")
        plt.xlabel("Epoch")
        plt.ylabel("Average loss")
        plt.savefig(self.name + "/convergence_deep_q_learning.png")
        # plt.show()
        plt.close()
        
        plt.plot(avg_len_path, 'b', label='len path')
        #plt.plot(epsilons, 'r', label='Exploration Rate')
        plt.title("Convergence of DQN Algorithm number action") 
        plt.xlabel("Epoch")
        plt.ylabel("Average len path")
        plt.savefig(self.name + "/convergence_deep_q_learning_len_path.png")
        plt.close()
        
        Policy = self.evaluation_q_network # The neural network trained is function that outputs the policy
        ########## Needs to be saved somewhere ##########
        return Policy
        #return -mean_length_path, losses
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)
            
    def learn(self, current_state_batch, action_batch, reward_batch, next_state_batch,  params_dict = {}):
        
        gamma =  params_dict["gamma"]
        max_norm = params_dict["max_norm"] 
        do_gradient_clipping = params_dict["do_gradient_clipping"]
        
        # print(f'max_norm: {max_norm}')
        # print(f'do_gradient_clipping: {do_gradient_clipping}')
        # print(f'type(max_norm): {type(max_norm)}')
        # print(f'type(do_gradient_clipping): {type(do_gradient_clipping)}')
        
        self.evaluation_q_network.train()
        self.target_q_network.eval() # So that we don't calculate the gradient of the Target Q Network

        # Convert batch data to PyTorch tensors
        current_state_tensor = torch.tensor(current_state_batch, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_state_batch, dtype=torch.float32)
        reward_tensor = torch.tensor(reward_batch, dtype=torch.float32)
        
        # Get Q-values for the current state-action pairs
        q_values_current = self.evaluation_q_network(current_state_tensor)
        
        # Use the target network to get Q-values for the next state for stability
        with torch.no_grad(): # Gradient calculation is disabled to save memory and speed up the process.
            q_values_next = self.target_q_network(next_state_tensor)
        
        # Get the maximum Q-value for each next state
        max_q_values_next = torch.max(q_values_next, dim=1)[0]
        
        # Calculate the target Q-values using the Bellman equation
        target_q_values = reward_tensor.squeeze() + (gamma * max_q_values_next)
        
        # Calculate the loss between the predicted and target Q-values
        # selected_q_values = torch.zeros([len(action_batch),1])
        # for i, action in enumerate(action_batch):
        #     selected_q_values[i] = q_values_current[i, action]
        
        action_batch = torch.tensor(action_batch, dtype=torch.long).squeeze()
        selected_q_values = q_values_current.gather(1, action_batch.unsqueeze(1)).squeeze()

        loss = self.evaluation_q_network.loss_fct(selected_q_values, target_q_values)
        
        # Backpropagation
        self.evaluation_q_network.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping by norm
        if do_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(self.evaluation_q_network.network.parameters(), max_norm)
        self.evaluation_q_network.optimizer.step()
        
        return loss.item()


def save_model_complexity(model, params_dict, filename="complexity_report.txt"):
    """Compute and save the model complexity and memory usage."""
    
    # Count model parameters
    n_params = sum(p.numel() for p in model.parameters())
    mem_MB = n_params * 4 / 1024**2  # FP32 weights
    mem_MB_total = mem_MB * 3        # include Adam optimizer states (approx)
    
    # Compute per-forward and per-batch ops
    input_dim = params_dict["hidden_layers"]
    hidden_layers = params_dict["params_list"]
    batch_size = params_dict["batch_size"]
    n_actions = params_dict["n_actions"]
    
    # Forward FLOPs (mult-adds)
    ops_forward = input_dim*hidden_layers[0] + sum(hidden_layers[i]*hidden_layers[i+1] for i in range(len(hidden_layers)-1)) + hidden_layers[-1]*n_actions
    
    ops_train_step = 4 * ops_forward                # forward + backward + target
    ops_per_batch = ops_train_step * batch_size
    
    # Replay buffer memory (assuming float32)
    buffer_size = params_dict.get("replay_buffer_memory_size", 80000)
    buffer_entry = (2 * input_dim + 2)  # (s, s', a, r)
    buffer_MB = buffer_size * buffer_entry * 4 / 1024**2
    
    # Write results
    report = f"""
    [MODEL COMPLEXITY REPORT]

    Network architecture: {input_dim} -> {' -> '.join(map(str, hidden_layers))} -> {n_actions}
    Total parameters: {n_params:,}
    Model memory (FP32): {mem_MB:.3f} MB
    Model + optimizer memory (≈×3): {mem_MB_total:.3f} MB
    Replay buffer: {buffer_size:,} samples × {buffer_entry} floats
    Replay buffer memory (FP32): {buffer_MB:.3f} MB

    Ops per forward pass: {ops_forward:,}
    Ops per training sample (≈4×forward): {ops_train_step:,}
    Ops per batch (batch={batch_size}): {ops_per_batch/1e6:.2f} million

    Total estimated training cost per epoch:
    ≈ n_time_steps × n_channels_train × max_len_path × ops_per_batch
    (computed analytically in paper)
    """
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(report)
    
    print(f"[INFO] Complexity report saved to {filename}")
