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

from dataclasses import dataclass

@dataclass
class ExperimentContext:
    """
    Container for all objects needed to run one experiment.
    Builder creates it, and Runner consumes it.
    """
    parameters: Parameters
    store: Store
    environment: Environment
    probability: Probability
    channel: Channel
    feedback: Feedback

class ExperimentBuilder:
    """ 
    Builds all experiments objects and returns an ExperimentContext..
    """
    def __init__(self, *, parameters: Parameters, store: Store):
        self.parameters = parameters
        self.store = store
    
    def build(self, dataset_mode=None, noise_mode=None) -> ExperimentContext:
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

        # Dataset mode 
        if dataset_mode is None:
            if self.parameters.use_sionna_dataset:
                if not self.parameters.sionna_dataset_pickle_path:
                    raise ValueError(
                        "use_sionna_dataset=True but sionna_dataset_pickle_path is not set."
                    )
                dataset_mode = DatasetMode.IMPORT_SIONNA
            else:
                dataset_mode = (
                    DatasetMode.LOAD_GENERATED
                    if self.parameters.continue_training
                    else DatasetMode.GENERATE
                )

        # Generate/import dataset
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

        ## Generate noise parameters (mean,std) 
        if noise_mode is None:
            if self.parameters.use_sionna_dataset:
                noise_mode = NoiseMode.ANALYTICAL
            else:
                noise_mode = (
                    NoiseMode.REUSE
                    if self.parameters.continue_training
                    else NoiseMode.FIT
                )

        if self.parameters.use_sionna_dataset and noise_mode == NoiseMode.FIT:
            raise ValueError(
                "NoiseMode.FIT is not supported with use_sionna_dataset=True. "
                "Use NoiseMode.ANALYTICAL for now. Change to NoiseMode.FIT_SIONNA if I want to fit noise parameters on the imported Sionna dataset."
            )

        noise_factory = NoiseFactory()

        noise_parameters = noise_factory.get_noise_params(
            parameters=self.parameters,
            noise_mode=noise_mode,
            store=self.store,
            feedback=feedback,
            channel=channel
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

        return ExperimentContext(
            parameters = self.parameters,
            store = self.store,
            environment = environment,
            probability = probability,
            channel = channel,
            feedback = feedback
        )