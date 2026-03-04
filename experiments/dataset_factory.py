from enum import Enum

from experiments.store import Store
from dataset.monteCarlo import Dataset_probability
from config.parameters import Parameters
from systemModel.channel import Channel
from systemModel.codebooks import Codebooks
from systemModel.feedback import Feedback

class DatasetMode(str, Enum):
    """ This mode defines how the dataset should be obtained for a given experiment.

    Values:
    ------
        GENERATE (str): Create a new dataset using the current experiment parameters and save it to the experiment directory. Use this mode when modifying channel/codebook parameters.
        
        REUSE (str): Load an existing dataset from disk. Use this mode when Re-running training on the same data to avoid expensive dataset regeneration.
    """
    GENERATE = "generate" 
    REUSE = "reuse"

class DatasetFactory:
    """ This class handles experiments dataset using the store by : 

        . Loading dataset from disk when DatasetMode.REUSE,
        . Generating dataset in memory when DatasetMode.GENERATE,
        . Saving generated dataset to the experiment folder.
    
    """
    
    def get_dataset(
            self,
            *,
            dataset_mode: DatasetMode,
            store: Store,
            parameters: Parameters,
            channel: Channel,
            codebooks: Codebooks,
            feedback: Feedback,
            noisy_samples: bool = True
        ) -> Dataset_probability:
        """ Main function responsible for returning the dataset for the experiment based on the specified mode.

            Args:
                dataset_mode (DatasetMode): The mode to determine how to obtain the dataset (GENERATE or REUSE).
                store (Store): The store instance to handle dataset loading and saving.
                parameters (Parameters): The experiment parameters.
                channel (Channel): The channel model for dataset generation.
                codebooks (Codebooks): The codebooks used in the experiment.
                feedback (Feedback): The feedback mechanism for dataset generation.
                noisy_samples (bool, optional): Whether to include noise in the generated samples. Defaults to True.
                
        ) -> Dataset_probability
        """
        if dataset_mode == DatasetMode.REUSE:
            if not store.dataset_exists():
                raise FileNotFoundError(f"No dataset found for reuse at {store.paths.dataset_pickle}. Please use a DatasetMode.GENERATE first.")
        
            print(f"[INFO] Reusing dataset from {store.paths.dataset_pickle}")
            
            return store.load_dataset()

        if dataset_mode == DatasetMode.GENERATE:
            print("[INFO] Generating new dataset")
            dataset = Dataset_probability(
                parameters = parameters, 
                channel = channel, 
                codebooks = codebooks, 
                feedback = feedback, 
                Noisy_samples= noisy_samples,
                filename = store.paths.root
                )
            store.save_dataset(dataset)
            print(f"[INFO] Dataset saved to {store.paths.dataset_pickle}")

            return dataset