import os, pickle
from typing import Tuple
from dataclasses import dataclass
from dataset.noise_calibration import load_fitted_noise

@dataclass(frozen=True) # with dataclass, Python automatically builds the constructor
                        # frozen so the root can not be modefied after creation
class ExperimentPaths:  
    root: str

    @property # allows to use a method like an attribute
    def dataset_pickle(self) -> str:
        return os.path.join(self.root, "Dataset.pickle")

    @property
    def noise_csv(self) -> str:
        return os.path.join(self.root, "Noise_parameters.csv")
    
class Store:
    def __init__(
            self, 
            paths: "ExperimentPaths"
    ):
        self.paths = paths
        os.makedirs(self.paths.root, exist_ok= True)

    # ---- Dataset ----
    def dataset_exists(self) -> bool:
        return os.path.exists(self.paths.dataset_pickle)
    
    def load_dataset(self):
        with open(self.paths.dataset_pickle, "rb") as f:
            return pickle.load(f)

    def save_dataset(self, dataset_obj) -> None:
        with open(self.paths.dataset_pickle, "wb") as f:
            pickle.dump(dataset_obj, f)

    # ---- Noise ----
    def noise_exists(self) -> bool:
        return os.path.exists(self.paths.noise_csv)
    
    def load_noise(self) -> Tuple[float, float]:
        mean, std = load_fitted_noise(self.paths.root)
        return (float(mean), float(std))