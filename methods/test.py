import os
import torch
import numpy as np 
from tqdm import tqdm
from matplotlib import pyplot as plt

from methods.methods import Methods
from systemModel.channel import Channel
from config.parameters import Parameters
from dataset.probability import Probability

from reinforcement_learning.q_learning.agent import QLearningAgent
from reinforcement_learning.deep_q_learning.agent import DeepQLearningAgent

class Test:
    """ This class is used for testing the Q-Learning and Deep Q-Learning Algorithms.
    """
    
    def __init__(   self,
                    parameters:Parameters,
                    methods: Methods,
                    channel:Channel,
                    probability: Probability,
                    DQN: DeepQLearningAgent
                    ) -> None:
    
        self.parameters = parameters
        self.methods = methods
        self.channel = channel
        self.probability = probability
        self.DQN = DQN

    def test(self, testing_objects_dict, epoch, Policy_network):
        
        ############## Parameters loop  ###############
        n_ex = 500 # Number of times we simulate
        T = 10 # Number of time steps for the evolution of the channel
        
        parameters = testing_objects_dict["parameters"]
        channel = testing_objects_dict["channel"]
        feedback = testing_objects_dict["feedback"]
        probability = testing_objects_dict["probability"]
        Hierarchical_possible = testing_objects_dict["Hierarchical_possible"]
        Policy_Q = testing_objects_dict["Policy_Q"]
        states = testing_objects_dict["states"]
        modification_channel = testing_objects_dict["modification_channel"]
        filename = testing_objects_dict["filename"]
        snr_values = testing_objects_dict["snr_values"]
        #Policy_network = self.evaluation_q_network
        
        if Hierarchical_possible:
            #label = ["Exhaustive","Hierarchical","Random sampling","Q-Learning Sampling","Boosted Q-Learning Probability","Boosted Q-Learning Highest"] ## Methods to compare
            label = ["Exhaustive","Hierarchical","Random sampling", "Deep Q-Learning","Q-Learning"] ## Methods to compare
            # label = ["Exhaustive","Hierarchical","Random sampling","Deep Q-Learning"] ## Methods to compare
            # label = ["Exhaustive","Hierarchical","Random sampling","Q-Learning"] ## Methods to compare
        else:
            label = ["Exhaustive","Random sampling","Q-Learning Sampling"] ## Methods to compare  
        
        for SNR in snr_values :  # Loop over all SNR values
            
            self.parameters.set_SNR(SNR)  # Ensure channel is updated for each SNR
            self.parameters.std_noise = math.sqrt(10**(-SNR/10))
            self.parameters.set_noise(mean_noise=0, std_noise=self.parameters.std_noise)
            print(f'SNR value = {self.parameters.SNR}')
            print(f'Std noise = {self.parameters.std_noise}')
            print(f'Noise parameters = {self.parameters.get_noise_parameters}')
            
            methods = Methods(  parameters,
                    channel,
                    feedback,
                    probability,
                    states,
                    Policy_Q,
                    Policy_network)
            
            correct_class = np.zeros((T,len(label)))
            Relative_strength = np.zeros((T,len(label)))
            Sum = np.zeros(T)
            
            stats_DQN_actions_list = []
            stats_QL_actions_list = []
            all_len_path_per_SNR = []  # List to store lengths of paths during testing per SNR
            
            total_successful_episodes = 0  # To count successful simulations
            successful_episodes_per_epoch = []  # To track successful episodes per epoch
            
            print(f'[INFO] Started testing for SNR = {self.parameters.SNR} and std noise = {self.parameters.std_noise}')
            
            for ex in tqdm(range(n_ex)):
                
                self.channel.new_channel() # New channel
                methods.forget() # We reinitialize the different algorithms
                
                #print("new")
                #print(channel.get_channel())
                
                successful_episode = False  # To check if there is at least one successful episode in this simulation
                total_len_path = T
                
                for t in range(0,T):
                    list_algorithms_class = [] # Stores the class that different methods output
                    ### To find the optimal codeword ###
                    index_optimal_cd = methods.optimal_codeword() #### Gives the index of the best codeword
                    #print("opt")
                    #print(index_optimal_cd)
                    
                    feedback.transmit(index_optimal_cd,codebook_used=0)
                    RSE_opt = feedback.get_feedback(noise = False)
            
                    ### Exhaustive search ###
                    index_exhaustive = methods.exhaustive() 
                    list_algorithms_class.append(index_exhaustive)
                    #print(index_exhaustive)
                    
                    ### Hierarchical search ###
                    if Hierarchical_possible:
                        index_hierarchical = methods.hierarchical()
                        list_algorithms_class.append(index_hierarchical)
                    #print(index_hierarchical)
                    
                    ### Random Sampling search ###
                    index_random_sampling = methods.random_sampling() 
                    list_algorithms_class.append(index_random_sampling)
                    #print(index_random_sampling)
                    
                    ### Deep Q-Learning search ###
                    index_deep_q_learning = methods.deep_q_learning_sampling()
                    list_algorithms_class.append(index_deep_q_learning)
                    stats_DQN_actions_list.append(index_deep_q_learning)
                    
                    ## Q-Learning search ###
                    index_q_learning = methods.q_learning_sampling()
                    list_algorithms_class.append(index_q_learning)
                    stats_QL_actions_list.append(index_q_learning)
                    
                    # prob_computed_random = methods.posterior_random
                    # if max(prob_computed_random) < 1-self.parameters.delta_init and not successful_episode:
                    #     len_path_computed += 1
                    # if max(prob_computed_random) >= 1-self.parameters.delta_init: 
                    #     successful_episode = True
                        
                    # prob_computed = methods.posterior_q
                    
                    #print(prob_computed)
                    #if max(prob_computed) < 1-delta_init and not successful_episode:
                        #len_path_computed += 1
                    #if max(prob_computed) >= 1-delta_init: 
                        #successful_episode = True
                    
                    ### Narrow beam search ###
                    #index_boost_proba = methods.test_narrow(index_optimal_cd)
                    #list_algorithms_class.append(index_boost_proba)
                    
                    ### Narrow beam search ###
                    #index_boost_highest = methods.test_narrow_2(index_optimal_cd)
                    #list_algorithms_class.append(index_boost_highest)
                    
                    #sum_prob = methods.test_maxprob(correct_class)
                    #Sum[t] = Sum[t] + sum_prob
                    
                    #print(t)
                    #if index_q_narrow == index_optimal_cd:
                        #print("Correct")
                    #if index_q_learning != index_optimal_cd:
                        #print("Incorrect")
                        
                    ### Metrics to measure performances ###
                    correct_class[t] = correct_class[t] + [index == index_optimal_cd for index in list_algorithms_class]
                    # CHANGE !!!!!!!!!!!!!!!!!!!!!!!!!!
                    # correct_class[t] = correct_class[t] + [feedback.transmit(index) == index_optimal_cd for index in list_algorithms_class]
                    for algo in range(0,len(list_algorithms_class)):
                        index = list_algorithms_class[algo]
                        feedback.transmit(index,codebook_used=0)
                        RSE = feedback.get_feedback(noise = False)
                        RSE_relative = RSE/RSE_opt
                        Relative_strength[t][algo] = Relative_strength[t][algo] + RSE_relative
                    
                    # Accumulate path lengths and rewards for QL
                    if index_q_learning == index_optimal_cd and not successful_episode:
                        successful_episode = True
                        total_len_path = t 

                    # Accumulate path lengths and rewards for DQN
                    if index_deep_q_learning == index_optimal_cd and not successful_episode:
                        successful_episode = True
                        total_len_path = t 
                        
                    channel.update(modification_channel = modification_channel) # The channel changes
                
                if successful_episode:
                    total_successful_episodes += 1
                
                successful_episodes_per_epoch.append(total_successful_episodes)
                
                # Store statistics for the current example
                all_len_path_per_SNR.append(total_len_path)
            
            # all_len_path.append(all_len_path_per_SNR)
            
            Sum = np.array(Sum)
            Sum = np.mean(Sum)
            
            correct_class /= n_ex
            Relative_strength /= n_ex
            
            # Convert lists to numpy arrays for easier manipulation
            all_len_path_per_SNR = np.array(all_len_path_per_SNR)
            
            # Calculate average values
            avg_len_path = np.mean(all_len_path_per_SNR)
            
            # Calculate standard deviation
            std_len_path = np.std(all_len_path_per_SNR)
            
            success_rate = (total_successful_episodes  /n_ex) * 100 
            
            # Save only the average path length to a .dat file for different SNRs
            output_data = {"Epoch": epoch, "AvgLenPath": avg_len_path}
            output_df = pd.DataFrame(output_data, index=[0])
            filename_data = os.path.join(filename, f"data_epoch_{epoch}_snr_{SNR}.dat")
            output_df.to_csv(filename_data, index=False)
            print(f"[INFO] Test data (AvgLenPath) saved to {filename_data}")

            # Save probability data for different methods for different SNRs
            prob_data = pd.DataFrame(correct_class, columns=label)
            prob_filename = os.path.join(filename, f"probability_epoch_{epoch}_snr_{SNR}.dat")
            prob_data.to_csv(prob_filename, index=False)
            print(f"[INFO] Probability data saved to {prob_filename}")
            
            # Plot Probability of Finding the Best Codeword
            plt.figure(figsize=(10, 6))
            plt.plot(np.arange(1, T + 1), correct_class, label=label)
            plt.legend()
            plt.title(f'Probability of Finding the Best Codeword for Different Number of Pilots Sent at SNR = {SNR}')
            plt.grid()
            plt.savefig(f"{filename}/Image_snr_{SNR}_epoch_{epoch}.png") 
            plt.close()
            
            # Save strength data
            strength_data = pd.DataFrame(Relative_strength, columns=label)
            strength_filename = os.path.join(filename, f"strength_epoch_{epoch}_snr_{SNR}.dat")
            strength_data.to_csv(strength_filename, index=False)
            print(f"[INFO] Strength data saved to {strength_filename}")
            
            # Plot Strength curves
            plt.figure(figsize=(10, 6))
            plt.plot(np.arange(1, T + 1), Relative_strength, label=label)
            plt.legend()
            plt.title(f'Strength for Different Number of Pilots Sent (SNR = {SNR})')
            plt.grid()
            plt.savefig(f"{filename}/Image_strength_epoch_{epoch}_snr_{SNR}.png")
            plt.close()

            # # Plot Histogram of Actions Taken by DQN
            # plt.figure(figsize=(10, 6))
            # plt.hist(stats_DQN_actions_list, bins=range(min(stats_DQN_actions_list), max(stats_DQN_actions_list) + 2), edgecolor='black', align='left')
            # plt.xlabel('Action')
            # plt.ylabel('Frequency')
            # plt.title(f'Histogram of Actions Taken by DQN at SNR = {SNR}')
            # plt.savefig(f"{filename}/stats_DQN_snr_{SNR}_epoch_{epoch}.png")
            # plt.close()
            
            # # Plot Histogram of Actions Taken by Q-Learning
            # plt.figure(figsize=(10, 6))
            # plt.hist(stats_QL_actions_list, bins=range(min(stats_QL_actions_list), max(stats_QL_actions_list) + 2), edgecolor='black', align='left')
            # plt.xlabel('Action')
            # plt.ylabel('Frequency')
            # plt.title(f'Histogram of Actions Taken by Q-Learning at SNR = {SNR}') 
            # plt.savefig(f"{filename}/stats_QL_actions_snr_{SNR}_epoch_{epoch}.png") 
            # plt.close()

            # # Plot Histogram of Path Lengths
            # plt.figure(figsize=(10, 6))
            # plt.hist(all_len_path_per_SNR, bins=20, alpha=0.5, label='Path Length')
            # plt.xlabel('Value')
            # plt.ylabel('Frequency')
            # plt.title(f'Distribution of Path Lengths at SNR = {SNR}')
            # plt.legend()
            # plt.savefig(f"{filename}/distribution_path_length_snr_{SNR}_epoch_{epoch}.png")
            # plt.close()

            # Plot Number of Successful Episodes by Epoch
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(successful_episodes_per_epoch) + 1), successful_episodes_per_epoch, marker='o', linestyle='-', color='b')
            plt.xlabel('Epoch')
            plt.ylabel('Number of Successful Episodes')
            plt.title(f'Number of Successful Episodes by Epoch at SNR = {SNR}')  
            plt.grid(True)
            plt.savefig(f"{filename}/successful_episodes_by_epoch_at_snr_{SNR}.png")  
            plt.close()

            # Print Statistics
            print(f"[INFO] When SNR = {SNR} :")
            print(f"[INFO] Success Rate: {success_rate}%")
            print(f'[INFO] Successful episodes: {total_successful_episodes} out of {n_ex}')
            print(f'[INFO] Average Path Length: {avg_len_path}')
            print(f'[INFO] Standard Deviation of Path Lengths: {std_len_path}')
        
    def run_tests(self, testing_objects_dict, model, checkpoints_dir, params_type):
        
        test_class = Test(  self.parameters,
                            self.methods,
                            self.channel,
                            self.probability,
                            self.DQN)
 
        # Save parameters once before the loop
        params_filename = os.path.join(checkpoints_dir, "params.txt")
        test_class.parameters.save_to_file(params_filename, params_type)
        print(f"[INFO] Parameters saved to {params_filename}")
        
        checkpoint_files = sorted([f for f in os.listdir(checkpoints_dir) if f.endswith('_eval.pth')])

        for epoch, checkpoint in enumerate(checkpoint_files):
            if epoch % self.parameters.test_freq == 0:
                checkpoint_path = os.path.join(checkpoints_dir, checkpoint)
                print(checkpoint_path)
                model.load_state_dict(torch.load(checkpoint_path))
                # model.load_state_dict(torch.load(checkpoint_path, weights_only=False))
                model.eval()
                print(f"[INFO] Testing model at epoch {epoch}")
                test_class.test(testing_objects_dict, epoch, model)


    def test_saved_dql_model(self, testing_objects_dict, checkpoints_dir, params_type):
        
        test_class = Test(  self.parameters,
                            self.methods,
                            self.channel,
                            self.probability,
                            self.DQN)

        # Save parameters once before the loop
        params_filename = os.path.join(checkpoints_dir, "params.txt")
        test_class.parameters.save_to_file(params_filename, params_type)
        print(f"[INFO] Parameters saved to {params_filename}")
        
        checkpoint_files = sorted([f for f in os.listdir(checkpoints_dir) if f.endswith('_eval.pth')])

        for checkpoint in checkpoint_files:
            # Extract the epoch number from the filename
            epoch_str = ''.join(filter(str.isdigit, checkpoint))
            if epoch_str.isdigit():
                epoch = int(epoch_str)
            else:
                continue  # If the filename doesn't contain a valid number, skip it

            if epoch % self.parameters.test_freq == 0:
                checkpoint_path = os.path.join(checkpoints_dir, checkpoint)
                print(checkpoint_path)

                # Instantiate the DQN model and load the state dict from the checkpoint
                model = DQN(self.parameters, input_dims=self.DQN.input_dims, n_actions=self.DQN.n_actions)
                model.load_state_dict(torch.load(checkpoint_path))
                model.eval()

                print(f"[INFO] Testing model at epoch {epoch}")
                test_class.test(testing_objects_dict, epoch, model)


    def test_saved_models(self, testing_objects_dict, checkpoints_dir_ql, checkpoints_dir_dql):
        """
        Test the saved models for both Q-Learning and Deep Q-Learning.
        
        Parameters:
        - testing_objects_dict: A dictionary of objects needed for testing.
        - checkpoints_dir: Directory where the checkpoints are stored.
        """
        filename = testing_objects_dict["filename"]
        
        # Instantiate the Test class for both Q-Learning and Deep Q-Learning
        test_class = Test(self.parameters, self.methods, self.channel, self.probability, self.DQN)
        
        # Save parameters for both methods
        params_filename_dqn = os.path.join(checkpoints_dir_dql, "params_dqn.txt")
        params_filename_ql = os.path.join(checkpoints_dir_ql, "params_ql.txt")
        
        # Save Q-Learning parameters
        test_class.parameters.save_to_file(params_filename_ql, 'ql')
        print(f"[INFO] Q-Learning parameters saved to {params_filename_ql}")
        
        # Save Deep Q-Learning parameters
        test_class.parameters.save_to_file(params_filename_dqn, 'dqn')
        print(f"[INFO] Deep Q-Learning parameters saved to {params_filename_dqn}")
        
        print("[INFO] Starting testing...")
        
        # Load all saved DQN model checkpoints
        checkpoint_files = sorted([f for f in os.listdir(checkpoints_dir_dql) if f.endswith('_eval.pth')])
        
        for checkpoint in checkpoint_files:
            # Extract the epoch number from the filename
            epoch_str = ''.join(filter(str.isdigit, checkpoint))
            if epoch_str.isdigit():
                epoch = int(epoch_str)
            else:
                continue  # If the filename doesn't contain a valid number, skip it

            if epoch % self.parameters.test_freq_DQN == 0:
                checkpoint_path = os.path.join(checkpoints_dir_dql, checkpoint)
                print(f"[INFO] Loading DQN model from {checkpoint_path}")

                # Instantiate the DQN model and load the state dict from the checkpoint
                model = DQN(self.parameters, input_dims=self.DQN.input_dims, n_actions=self.DQN.n_actions)
                model.load_state_dict(torch.load(checkpoint_path))
                model.eval()

                # Perform testing for the current checkpoint
                print(f"[INFO] Testing DQN model at epoch {epoch}")
                test_class.test(testing_objects_dict, epoch, model)
        
        print("[INFO] Testing complete.\n")
    
    def test_saved_QL_models(self, testing_objects_dict, checkpoints_dir_ql):
        """
        Test the saved models of Q-Learning 
        """
        
        # Instantiate the Test class for Q-Learning 
        test_class = Test(self.parameters, self.methods, self.channel, self.probability, self.DQN)
        
        # Save parameters 
        params_filename_ql = os.path.join(checkpoints_dir_ql, "params_ql.txt")
        
        # Save Q-Learning parameters
        test_class.parameters.save_to_file(params_filename_ql, 'ql')
        print(f"[INFO] Q-Learning parameters saved to {params_filename_ql}")
        
        print("[INFO] Starting testing...")
        
        # Set the testing frequency (same for both Q-Learning and DQN)
        test_freq = self.parameters.test_freq_DQN
        
        # Load all saved Q-Learning policy files
        policy_files = sorted([f for f in os.listdir(checkpoints_dir_ql) if f.startswith('policy_after_') and f.endswith('.csv')])
        
        for policy_file in policy_files:
            # Extract the epoch number from the filename
            epoch_str = ''.join(filter(str.isdigit, policy_file.split('_after_')[-1].split('episodes')[0]))
            if epoch_str.isdigit():
                epoch = int(epoch_str)
            else:
                continue  # If the filename doesn't contain a valid number, skip it
            
            if epoch % test_freq == 0:
                # Load the Q-Learning policy
                policy_path = os.path.join(checkpoints_dir_ql, policy_file)
                print(f"[INFO] Loading Q-Learning policy from {policy_path}")
                policy = self.load_Policy(epoch, checkpoints_dir_ql)  # Use your load_Policy method
                
                # Set the loaded policy in the Q-Learning class
                self.policy = policy  
                
                # Perform testing for both Q-Learning and DQN
                print(f"[INFO] Testing at epoch {epoch}")
                test_class.test(testing_objects_dict, epoch, Policy_network=None, filename=checkpoints_dir_ql)
    
    print("[INFO] Testing complete.\n")
    
    def test_QL(self, testing_objects_dict, filename, epochs):
        
        parameters = testing_objects_dict["parameters"]
        channel = testing_objects_dict["channel"]
        feedback = testing_objects_dict["feedback"]
        probability = testing_objects_dict["probability"]
        Hierarchical_possible = testing_objects_dict["Hierarchical_possible"]
        states = testing_objects_dict["states"]
        filename = testing_objects_dict["filename"]
        
        ##############
        n_ex = 10 # Number of times we simulate
        T = 10 # Number of time steps for the evolution of the channel
        ##############

        print(f'[INFO] Started testing...')
        all_len_path = []  # List to store lengths of paths during testing
        all_correct = []
        all_strength = []
        
        for _ in range(epochs):
            Policy = load_Policy(epochs,filename+"/Q_matrices")

            methods = Methods(  parameters,
                                channel,
                                feedback,
                                probability,
                                states,
                                Policy)


            Hierarchical_possible = True

            if Hierarchical_possible:
                    #label = ["Exhaustive","Hierarchical","Random sampling","Q-Learning Sampling","Boosted Q-Learning Probability","Boosted Q-Learning Highest"] ## Methods to compare
                    # label = ["Exhaustive","Hierarchical","Random sampling", "Deep Q-Learning","Q-Learning"] ## Methods to compare
                    # label = ["Exhaustive","Hierarchical","Random sampling","Deep Q-Learning"] ## Methods to compare
                    label = ["Exhaustive","Hierarchical","Random sampling","Q-Learning"] ## Methods to compare
            else:
                    label = ["Exhaustive","Random sampling","Q-Learning Sampling"] ## Methods to compare  
            
            correct_class = np.zeros((T,len(label)))
            Relative_strength = np.zeros((T,len(label)))

            stats_QL_actions_list = []

            total_len_path = 0

            for ex in tqdm(range(n_ex)):
                channel.new_channel() # New channel
                methods.forget() # We reinitialize the different algorithms   
                #print("new")
                #print(channel.get_channel()) 
                successful_episode = False  # To check if there is at least one successful episode in this simulation
                len_path_computed = 1 # Starts at 1
                for t in range(0,T):
                    list_algorithms_class = [] # Stores the class that different methods output
                    ### To find the optimal codeword ###
                    index_optimal_cd = methods.optimal_codeword() #### Gives the index of the best codeword
                    
                    feedback.transmit(index_optimal_cd,codebook_used=0)
                    RSE_opt = feedback.get_feedback(noise = False)
                    ### Exhaustive search ###
                    index_exhaustive = methods.exhaustive() 
                    list_algorithms_class.append(index_exhaustive)
                    #print(index_exhaustive)
                    
                    ### Hierarchical search ###
                    if Hierarchical_possible:
                        index_hierarchical = methods.hierarchical()
                        list_algorithms_class.append(index_hierarchical)
                    #print(index_hierarchical)
                    
                    ### Random Sampling search ###
                    index_random_sampling = methods.random_sampling() 
                    list_algorithms_class.append(index_random_sampling)
                    #print(index_random_sampling)
                    
                    ### Q-Learning search ###
                    index_q_learning = methods.q_learning_sampling()
                    list_algorithms_class.append(index_q_learning)
                    stats_QL_actions_list.append(index_q_learning)
                    
                    prob_computed = methods.posterior_q
                    if max(prob_computed) < 1-parameters.delta_init:
                        len_path_computed += 1
                        
                    ### Metrics to measure performances ###
                    correct_class[t] = correct_class[t] + [index == index_optimal_cd for index in list_algorithms_class]
                    for algo in range(0,len(list_algorithms_class)):
                        index = list_algorithms_class[algo]
                        feedback.transmit(index,codebook_used=0)
                        RSE = feedback.get_feedback(noise = False)
                        RSE_relative = RSE/RSE_opt
                        Relative_strength[t][algo] = Relative_strength[t][algo] + RSE_relative
                    
                    channel.update() # The channel changes
                    
                total_len_path = total_len_path + len_path_computed
                
                # Store statistics for the current example
            all_len_path.append(total_len_path/n_ex)
            all_correct.append(correct_class/n_ex) #Average on channels   
            all_strength.append(Relative_strength/n_ex)

        # Convert lists to numpy arrays for easier manipulation
        all_len_path = np.array(all_len_path)
        all_correct = np.array(all_correct)
        all_strength = np.array(all_strength)

        # Save all the average path length for the fixed epoch to a .dat file
        output_data = {"Epoch": np.arange(1, epochs+1), "AvgLenPath": all_len_path}
        output_df = pd.DataFrame(output_data)
        filename_data = os.path.join(filename, f"data_epoch_{epochs}.dat")
        output_df.to_csv(filename_data, index=False)
        print(f"[INFO] Test data (AvgLenPath) saved to {filename_data}")

        # Plot Lstar for all epochs
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, epochs+1), all_len_path, marker='o')  # Plot against the range of epochs
        plt.title("L star")
        plt.grid()
        plt.savefig(filename + f"/Image_Lstar_epoch_{epochs}")
        plt.close()

        # Save probability data for different methods
        prob_data = pd.DataFrame(all_correct[0], columns=label)
        prob_filename = os.path.join(filename, f"probability_epoch_{epochs}.dat")
        prob_data.to_csv(prob_filename, index=False)
        print(f"[INFO] Probability data saved to {prob_filename}")

        # Plot Probability of Finding the Best Codeword
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, T + 1), all_correct[0], label=label)
        plt.legend()
        plt.title("Probability of Finding the Best Codeword for Different Number of Pilots Sent")
        plt.grid()
        plt.savefig(filename + f"/Image_proba_epoch_{epochs}")
        plt.close()

        # Save strength data for different methods
        strength_data = pd.DataFrame(all_strength[0], columns=label)
        strength_filename = os.path.join(filename, f"strength_epoch_{epochs}.dat")
        strength_data.to_csv(strength_filename, index=False)
        print(f"[INFO] Strength data saved to {strength_filename}")

        # Plot strength
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, T + 1), all_strength[0], label=label)
        plt.legend()
        plt.title("Strength for Different Number of Pilots Sent")
        plt.grid()
        plt.savefig(filename + f"/Image_strength_epoch_{epochs}")
        plt.close()
