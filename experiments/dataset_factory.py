import pickle
from enum import Enum

from experiments.store import ExperimentPaths
from dataset.monteCarlo import Dataset_probability

class DatasetMode(str, Enum):
    """ This mode defines how the dataset should be obtained for a given experiment.

    Values:
    ------
        GENERATE (str): Create a new dataset using the current experiment parameters and save it to the experiment directory. Use this mode when modifying channel/codebook parameters.
        
        REUSE (str): Load an existing dataset from disk. Use this mode when Re-running training on the same data, performing additional evaluations to avoid expensive dataset regeneration.
    """
    GENERATE = "generate" 
    REUSE = "reuse"

class DatasetFactory:
    """ This class handles experiments dataset by : 

        . Loading dataset from disk when DatasetMode.REUSE,
        . Generating dataset in memory when DatasetMode.GENERATE,
        . Saving generated dataset to the experiment folder.
    
    """
    def load(self, *, paths: ExperimentPaths) -> Dataset_probability:
        with open(paths.dataset_path, "rb") as f:
            return pickle.load(f)
        
    def save(self, *, paths: ExperimentPaths, dataset: Dataset_probability) -> None:
        with open(paths.dataset_path, "wb") as f:
            pickle.dump(dataset, f, protocol= pickle.HIGHEST_PROTOCOL)
    
    def get_dataset(
            self,
            *,
            dataset_mode: DatasetMode,
            paths: ExperimentPaths,
            
    ) -> Dataset_Probability