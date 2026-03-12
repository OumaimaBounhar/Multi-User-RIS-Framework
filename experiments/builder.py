from experiments.store import Store
from config.parameters import Parameters

from systemModel.channel import Channel
from systemModel.signal import Signal
from systemModel.codebooks import Codebooks
from systemModel.feedback import Feedback

from experiments.dataset_factory import DatasetFactory, DatasetMode
from experiments.noise_factory import NoiseFactory, NoiseMode
from dataset.probability import Probability
from reinforcement_learning.env import Environment
from reinforcement_learning.states import State

from experiments.runner import Runner

class ExperimentBuilder:
    """ 
    Builds all experiments objects and returns a ready-to-run Runner.
    """
    def __init__(self, *, parameters: Parameters, store: Store):
        self.parameters = parameters
        self.store = store
    
    def build(self) -> Runner:
        # System model

        # The channel model
        channel = Channel(
            self.parameters
        ) 
        
        # The transmitted signal
        signal = Signal(
            self.parameters
        ) 
        
        # The codebook for the RIS
        codebooks = Codebooks(
            self.parameters,
            seed=self.parameters.experiment_seed
        ) 

        # The feedback function between the receiver and the transmitter
        feedback = Feedback(
            self.parameters,
            channel,
            codebooks,
            signal
            ) 

        dataset_mode = (
            DatasetMode.REUSE
            if self.parameters.continue_training
            else DatasetMode.GENERATE
        )

        # Generate dataset
        dataset_factory = DatasetFactory()
        dataset_proba = dataset_factory.get_dataset(
            dataset_mode = dataset_mode, 
            store = self.store, 
            parameters = self.parameters, 
            channel = channel, 
            codebooks = codebooks, 
            feedback = feedback,
            noisy_samples = True
            )

        # Generate noise parameters (mean,std) 
        noise_factory = NoiseFactory()
        noise_parameters = noise_factory.get_noise_params( 
            parameters = self.parameters,
            noise_mode=NoiseMode.FIT, 
            store = self.store,
            feedback = feedback,
            channel = channel 
            )

        # Probability
        probability = Probability(
            self.parameters, 
            dataset_proba, 
            noise_parameters
            )

        # States space 
        states = State(
            self.parameters
            )

        # RL Environment
        environment = Environment(
            states=states,
            parameters=self.parameters,
            probability=probability,
            dataset_train=dataset_proba
            )

        return Runner(
            parameters= self.parameters,
            environment= environment, 
            store= self.store, 
            probability= probability,
            channel=channel,
            feedback=feedback
            )