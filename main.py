import numpy as np
import os

from config.parameters import Parameters

from dataset.monteCarlo import Dataset_probability
from dataset.probability import Probability

from systemModel.channel import Channel
from systemModel.codebooks import Codebooks
from systemModel.feedback import Feedback
from systemModel.signal import Signal

from methods.test import Test

from reinforcement_learning.env import Environment
from reinforcement_learning.states import State
from reinforcement_learning.q_learning.agent import QLearningAgent
from reinforcement_learning.deep_q_learning.agent import DeepQLearningAgent
from reinforcement_learning.deep_q_learning.components.seed import set_seed


########################################################################### 
########################## Instantiate Objects ############################
###########################################################################

all_size_of_codebooks = [
                        [8, 14],
                        # [16, 30],
                        # [32, 62],
                        # [64, 126]
                        ]

all_type_of_codebooks = [
                        ["Narrow_8", "Hierarchical_3_2"],
                        # ["Narrow_16", "Hierarchical_4_2"],
                        # ["Narrow_32", "Hierarchical_5_2"],
                        # ["Narrow_64", "Hierarchical_6_2"]
                        ]

delta_values = [
                # 3e-1,
                # 2e-1,
                1e-2,
                # 2e-2
                ]

for size_codebooks, type_codebooks in zip(all_size_of_codebooks, all_type_of_codebooks):
    for delta in delta_values:
        print("="*80)
        print(f"[INFO] Starting simulation for codebooks {type_codebooks} with Δ={delta}")
        print("="*80)

        check_size_cd(type_codebooks,size_codebooks) 

        parameters = Parameters(    N_R=64, 
                                    N_T=1, 
                                    N_RIS=100, 
                                    size_codebooks=size_codebooks, 
                                    type_codebooks=type_codebooks,
                                    mean_noise=0,
                                    SNR=10,
                                    # snr_values = [0,5,10,20],
                                    snr_values = [20],
                                    modification_channel=0,
                                    type_channel="half-spaced ULAs",
                                    type_modulation="BPSK", 
                                    mean_channel=0, 
                                    std_channel=[], 
                                    sigma_alpha=0, 
                                    gamma=0.94,
                                    learning_rate_init=5e-5,
                                    params_list=[256,256],
                                    batch_size=1024,
                                    replay_buffer_memory_size=120000,
                                    max_norm=0.5,
                                    do_gradient_clipping = True,
                                    loss_fct='mse',
                                    n_epochs=10000,
                                    # n_epochs = 5,
                                    n_time_steps_dqn=64,
                                    n_channels_train_DQN=5,
                                    # n_channels_train_DQN=1,
                                    freq_update_target=20,
                                    tau = 0.05,
                                    max_len_path=20,
                                    epsilon=1,
                                    epsilon_decay=0.992,
                                    epsilon_min=0.01,
                                    delta_init=delta,
                                    delta_decay=1,
                                    train_or_test=True,
                                    Train_Deep_Q_Learning=True,
                                    saving_freq_DQN=500,
                                    # saving_freq_DQN=1,
                                    test_freq_DQN=1,
                                    Train_Q_Learning=True,
                                    n_episodes=10000,
                                    # n_episodes=5,
                                    n_time_steps_ql=64,
                                    n_channels_train_QL=5,
                                    len_path=20,
                                    saving_freq_QL = 500,
                                    # saving_freq_QL = 1,
                                    test_freq_QL = 1,
                                    delta_final=5e-2, 
                                    len_window_action=1, 
                                    len_window_channel=10, 
                                    precision=2,  
                                    blabla_other_states=1, 
                                    min_representatives_q_learning_train=100, 
                                    min_representatives_q_learning_test=10,
                                    learning_rate_decay = 0.99,
                                    learning_rate_min = 1e-4
                                    
                                    ) # All the parameters stored in a class

        channel = Channel(parameters) # The channel model

        signal = Signal(parameters) # The signal sent

        codebooks = Codebooks(parameters) # The codebook for the RIS

        feedback = Feedback(parameters,
                            channel,
                            codebooks,
                            signal) # The feedback function between the receiver and the transmitter

        ##########################################
        ############## Saved Data  ###############
        ##########################################

        ### To decide if we reuse the dataset already generated or create a new one ###
        # number_exp = str(input("Where do you want to save the file? (number): "))

        #old_xp_names = os.listdir('/home/infres/obounhar-23/ris/Code/RIS/RIS/Data/')

        old_xp_names = os.listdir('./Data/')

        indices = [int(xp_name.split('_')[1]) for xp_name in old_xp_names]

        number_exp = max(indices) + 1
        print(f'using number_exp: {number_exp}')

        filename = "./Data/Example_{}".format(number_exp)  ## Name of the file where the dataset was saved
        if not os.path.exists(filename):
            os.makedirs(filename)
            
        analytical_expression_noise = True # If the Real noise of the data has a known expression
        Noisy_samples = True # If the samples from the dataset are noisy

        new_fit_noise = False ## Fit the real noise onto gaussian noise
        new_dataset = True ## New dataset if True else take the one saved in filename

        ### For the noise we load if already fitted ###
        ### Fitting is quick ###
        if analytical_expression_noise:
            print("analytical_expression Not done")
            params_modeled_noise = 0,0.01
        elif not new_fit_noise:
            params_modeled_noise = load_fitted_noise(filename)
        else:
            params_modeled_noise = fit_noise(filename,feedback,channel,parameters)
            
        ### For the dataset we load if already generated ###
        ### Creating a dataset takes time ###
        if not new_dataset:
            dataset_proba  = pickle.load(open(filename+"/Dataset.pickle", "rb", -1))
            parameters,codebooks = dataset_proba.get_params_codebook()
            feedback = Feedback(parameters,channel,codebooks,signal)
        else: 
            dataset_proba = Dataset_probability(parameters,
                                                channel,
                                                codebooks,
                                                feedback,
                                                Noisy_samples=Noisy_samples,
                                                filename=filename)
            
        probability = Probability(  parameters,
                                    dataset_proba,
                                    params_modeled_noise) ## The probability we will use later

        ########################################################################### 
        ############################## Q-Learning ################################# 
        ###########################################################################
        q_learning_parameters = parameters.get_q_learning_parameters()

        # States space 
        states = State(parameters)

        if parameters.Train_Q_Learning:
            # Environment initialization
            dataset_Q_learning_train = dataset_proba
                    
            dataset_Q_learning_test = dataset_Q_learning_train

            environment = Environment(  states,
                                        parameters, 
                                        probability,
                                        dataset_Q_learning_train,
                                        dataset_Q_learning_test)

            # Q-Learning Agent
            agent = QLearningAgent( environment=environment,
                                    parameters = parameters,
                                    name_file=filename+"/Q_matrices")

            # Start training
            agent.train(params_dict = q_learning_parameters)
            
        Policy = load_Policy(parameters.n_episodes,filename+"/Q_matrices")

        # name = filename+f"/Q_matrices/frequency_after_{parameters.n_episodes}episodes.csv"
        name = filename+f"/Q_matrices/frequency_after_{parameters.n_episodes}episodes_with_delta_=_{parameters.delta_init}_and_alpha=_{parameters.learning_rate_init}.csv"

        with open(name, mode='r') as file:
            reader = csv.reader(file)
            #Policy = [[int(item) for item in row] for row in reader]
            freq = [float(row[0]) for row in reader]
        print([states.get_state_from_index((np.argsort(np.array(freq))[::-1])[i])for i in range(0,10)])

        methods = Methods(  parameters,
                            channel,
                            feedback,
                            probability,
                            states,
                            Policy)

        other_params = methods.get_parameters()

        len_window_channel = other_params["len_window_channel"]
        len_window_action = other_params["len_window_action"]
        Hierarchical_possible = other_params["Hierarchical_possible"]
        Policy = other_params["Policy_Q"]
        modification_channel = parameters.modification_channel

        testing_objects_dict = {
            "parameters": parameters,
            "channel": channel,
            "feedback": feedback,
            "probability": probability,
            "states": states,
            "filename" : filename,
            "len_window_channel": len_window_channel,
            "len_window_action": len_window_action,
            "Hierarchical_possible": Hierarchical_possible,
            "Policy_Q": Policy,
            "modification_channel" : modification_channel,
            "snr_values" : parameters.snr_values
        }

        ###########################################################################
        ############################## Deep-Q-Learning ############################
        ###########################################################################

        dqn_parameters = parameters.get_dqn_parameters()

        if parameters.Train_Deep_Q_Learning:
            
            #input_dims = environment.state_space.get_n_states() # Non c'est pas la taille du state space, c'est la taille des éléments du state space
            input_dims = environment.get_size_states() #The dimension of inputs for the Deep Neural network
            
            # Deep-Q-Learning Agent
            agent = DeepQLearningAgent( input_dims = input_dims, 
                                        environment =  environment,
                                        parameters= parameters,
                                        name_file = filename
                                    )

            # Start training
            Policy_network = agent.train(params_dict = dqn_parameters, testing_objects_dict = testing_objects_dict)
            
            # flops, macs, params = calculate_flops_hf(model_name=Policy_network, input_shape=(parameters.batch_size, 14))
            # print("%s FLOPs:%s  MACs:%s  Params:%s \n" %(Policy_network, flops, macs, params))
            
            # agent.save_model_complexity(Policy_network, params_dict = dqn_parameters)

        Policy_network.eval()

        ########################################################################### 
        ########################## Start Simulation ############################### 
        ###########################################################################

        test_model = Test(  parameters,
                            methods,
                            channel,
                            probability,
                            agent)
        # Directory where the checkpoints are saved
        checkpoints_dir_ql = f"{filename}/Q_matrices"
        checkpoints_dir_dql = f"{filename}/checkpoints"
        print(filename)

        # Run the tests
        test_model.test_saved_models(testing_objects_dict, checkpoints_dir_ql, checkpoints_dir_dql)
