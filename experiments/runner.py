import os
from typing import Dict, Any

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
        q_matrices_path = self.store.paths.q_matrices_dir
        os.makedirs(q_matrices_path, exist_ok=True)

        agent = QLearningAgent(
            environment=self.env,
            parameters=self.parameters,
            paths=self.store.paths
        )

        agent.train()
        
        # Load the newly trained policy
        return load_Policy(
            paths = self.store.paths,
            episode = self.parameters.n_episodes
        )
    
    def run_deep_q_learning(self):
        """This method initializes, trains and returns the policy for Deep Q-Learning.
        """
        if not self.parameters.Train_Deep_Q_Learning:
            print("[INFO] Skipping Deep Q-Learning training.")
            return None

        print("[INFO] Starting Deep Q-Learning Training...")

        agent = DeepQLearningAgent(
            parameters = self.parameters,
            environment = self.env,
            paths = self.store.paths
        )

        # Returns the trained network
        policy_network = agent.train()
        
        policy_network.eval()
        return agent, policy_network

    def _build_testing_objects_dict(self, q_policy = None) -> Dict[str, Any]:
        """This method builds the dictionary of testing objects that will be used during the training of Deep Q-Learning.
        """
        return {
                "parameters": self.parameters,
                "channel": self.channel, # Derived from Environment
                "feedback": self.feedback,
                "probability": self.probability,
                "states": self.env.state_space,
                "paths": self.store.paths,
                "len_window_channel": self.parameters.len_window_channel,
                "Policy_Q": q_policy,
                "modification_channel": self.parameters.modification_channel,
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

        # Rebuild the final testing dictionary with available policies
        testing_objects_dict = self._build_testing_objects_dict(q_policy=q_policy)
        
        # Initialize the Test class
        tester = Test(
            parameters = self.parameters,
            methods = methods,
            channel = self.channel,
            probability = self.probability,
            DQN = dqn_agent
        )

        # Choosing the mode dynamically based on available policies
        if q_policy is not None and dqn_agent is not None and dqn_policy_net is not None:
            mode = "both"
        elif dqn_agent is not None and dqn_policy_net is not None:
            mode = "dqn"
        elif q_policy is not None:
            mode = "ql"
        else:
            raise ValueError("No trained policy available for testing.")

        # Execute the multi-SNR testing suite
        tester.run_model_tests(
            testing_objects_dict=testing_objects_dict,
            checkpoints_dir_ql=self.store.paths.q_matrices_dir,
            checkpoints_dir_dql=self.store.paths.dqn_checkpoints_dir,
            mode=mode
        )
