import os
import csv
import numpy as np
import torch
from typing import dict, Any

from experiments.store import Store
from config.parameters import Parameters
from dataset.probability import Probability
from systemModel.channel import Channel
from systemModel.feedback import Feedback

from reinforcement_learning.env import Environment
from reinforcement_learning.q_learning.agent import QLearningAgent
from reinforcement_learning.q_learning.utils import load_Policy
from reinforcement_learning.deep_q_learning.agent import DeepQLearningAgent
from methods.methods import Methods
from methods.test import Test

class Runner:
    """ This class is responsible for running the experiments of Q-Learning and Deep Q-Learning. It executes the training and testing loops.
    """
    def __init__(self, *, parameters: Parameters, environment: Environment, store: Store, probability: Probability, channel: Channel, feedback: Feedback):
        self.parameters = parameters
        self.env = environment
        self.store = store
        self.probability = probability
        self.channel = channel
        self.feedback = feedback
        self.filename = store.paths.root

    def run_q_learning(self):
        """This method initializes, trains and returns the policy for Q-Learning.
        """
        if not self.parameters.Train_Q_Learning:
            print("[Info] Skipping Q-Learning training.")
            return None
        
        print("[INFO] Starting Q-Learning Training...")
        q_matrices_path = os.path.join(self.filename, "Q_matrices")
        os.makedirs(q_matrices_path, exist_ok=True)

        agent = QLearningAgent(
            environment=self.env,
            parameters=self.parameters,
            name_file=q_matrices_path
        )

        q_learning_params = self.parameters.get_q_learning_parameters()
        agent.train(params_dict = q_learning_params)
        
        # Load the newly trained policy
        return load_Policy(self.parameters.n_episodes, q_matrices_path)
    
    def run_deep_q_learning(self):
        """This method initializes, trains and returns the policy for Q-Learning.
        """
        if not self.parameters.Train_Deep_Q_Learning:
            print("[INFO] Skipping Deep Q-Learning training.")
            return None

        print("[INFO] Starting Deep Q-Learning Training...")

        dqn_params = self.parameters.get_dqn_parameters()
        testing_objects_dict = self._build_testing_objects_dict()
        input_dims = self.env.get_size_states()

        agent = DeepQLearningAgent(
            input_dims = input_dims,
            environment = self.env,
            parameters = self.parameters,
            name_file = self.filename
        )

        # Returns the trained network
        policy_network = agent.train(
            params_dict = dqn_params, 
            testing_objects_dict = testing_objects_dict
        )
        
        policy_network.eval()
        return agent, policy_network

    def _build_testing_objects_dict(self, q_policy = None) -> Dict[str, Any]:
        """This method builds the dictionary of testing objects that will be used during the training of Deep Q-Learning.
        """
        return {
                "parameters": self.parameters,
                "channel": self.env.parameters, # Derived from Environment
                "feedback": self.env.probability,
                "probability": self.probability,
                "states": self.env.state_space,
                "filename": self.filename,
                "Policy_Q": q_policy,
                "snr_values": self.parameters.snr_values
            }

    def run_testing(self, q_policy=None, dqn_agent=None, dqn_policy_net=None):
        """
        Test both models (QL and DQL) to compare them to the state of the art.
        """
        print("[INFO] Starting Final Evaluation Simulation...")

        # Initialize Methods with current policies for comparative testing
        methods = Methods(
            parameters = self.parameters,
            channel = self.channel, 
            feedback = self.feedback,
            probability = self.probability,
            state = self.env.state_space,
            Policy_Q = q_policy,
            Policy_network = dqn_policy_net
        )

        # 2. Rebuild the final testing dictionary with available policies
        testing_objects_dict = self._build_testing_objects_dict(q_policy=q_policy)
        
        # 3. Initialize the Test class
        tester = Test(
            parameters = self.parameters,
            methods = methods,
            channel = self.channel,
            probability = self.probability,
            DQN = dqn_agent
        )

        # 4. Execute the multi-SNR testing suite
        checkpoints_dir_ql = os.path.join(self.filename, "Q_matrices")
        checkpoints_dir_dql = os.path.join(self.filename, "checkpoints")

        tester.test_saved_models(
            testing_objects_dict=testing_objects_dict,
            checkpoints_dir_ql=checkpoints_dir_ql,
            checkpoints_dir_dql=checkpoints_dir_dql
        )