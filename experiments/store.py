import os, pickle
import pandas as pd
from typing import Tuple
from dataclasses import dataclass

@dataclass(frozen=True) # with dataclass, Python automatically builds the constructor
                        # frozen so the root can not be modefied after creation
class ExperimentPaths:  
    root: str

    @classmethod 
    def make_new_experiment_folder(cls, base_dir: str = "./Data/") -> "ExperimentPaths":
        """Determines the next Example_N index without creating the directory."""
        if not os.path.exists(base_dir):
            return cls(root=os.path.join(base_dir, "Example_0"))
        
        existing_folders = [
            d for d in os.listdir(base_dir) 
            if d.startswith("Example_") and os.path.isdir(os.path.join(base_dir, d))
        ]

        indices = [
            int(d.split("_")[1]) for d in existing_folders 
            if len(d.split("_")) > 1 and d.split("_")[1].isdigit()
        ]
        
        # Changed to 0 for consistent indexing
        next_idx = max(indices) + 1 if indices else 0 

        return cls(root=os.path.join(base_dir, f"Example_{next_idx}"))
    
    @property # allows to use a method like an attribute
    def dataset_pickle(self) -> str:
        return os.path.join(self.root, "Dataset.pickle")

    @property
    def noise_csv(self) -> str:
        return os.path.join(self.root, "Noise_parameters.csv")
    
class Store:
    """
    Data Access Layer for experiment I/O and directory management.
    """
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
            pickle.dump(dataset_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ---- Noise ----
    def noise_exists(self) -> bool:
        return os.path.exists(self.paths.noise_csv)

    def save_noise(self, mean: float, std: float) -> None:
        """Saves noise parameters to a CSV file."""
        df = pd.DataFrame({"Mean": [mean], "Std": [std]})
        df.to_csv(self.paths.noise_csv, index=False)
    
    def load_noise(self) -> Tuple[float, float]:
        df = pd.read_csv(self.paths.noise_csv)
        return float(df["Mean"].iloc[0]), float(df["Std"].iloc[0])