import os, pickle
import pandas as pd
from typing import Tuple, Union
from dataclasses import dataclass

@dataclass(frozen=True) # with dataclass, Python automatically builds the constructor
                        # frozen so the root can not be modefied after creation
class ExperimentPaths:  
    root: str

    @classmethod 
    def make_new_experiment_folder(cls, base_dir: str = "./Data/") -> "ExperimentPaths":
        """Determines the next Example_N index without creating the directory."""
        if not os.path.exists(base_dir):
            return cls(root = os.path.join(
                base_dir, 
                "Example_0"
                )
            )
        
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

        return cls(root = os.path.join(
            base_dir, 
            f"Example_{next_idx}"
            )
        )
    
    @property # allows to use a method like an attribute
    def dataset_pickle(self) -> str:
        return os.path.join(
            self.root, 
            "Dataset.pickle"
        )

    @property
    def noise_csv(self) -> str:
        return os.path.join(
            self.root, 
            "Noise_parameters.csv"
        )

    # ---------------------------------------- Q-Learning  ------------------------------------------------------------------------------
    @property
    def q_matrices_dir(self) -> str:
        return os.path.join(
            self.root, 
            "Q_matrices"
        )

    def q_matrix_file(self, episode: int, is_last: bool = False) -> str:
        """ Save the Q-matrix as a CSV file
        Args:
            episode (int): The episode number after which the Q-matrix is saved.
            is_last (bool): Whether this is the final Q-matrix after training completion. If True, the filename will indicate it's the last one.
        Returns:
            str: The path to save the Q-matrix CSV file.
        """
        prefix = "Q_matrix_final_after_" if is_last else f"Q_matrix_after_"
        return os.path.join(
            self.q_matrices_dir, 
            f"{prefix}{episode}episodes.csv"
        )

    def policy_matrix_file(self, episode: int, is_last: bool = False) -> str:
        """ Save the policy matrix as a CSV file
        Args:
            episode (int): The episode number after which the policy matrix is saved.
            is_last (bool): Whether this is the final policy matrix after training completion. If True, the filename will indicate it's the last one.
        Returns:
            str: The path to save the policy matrix CSV file.
        """
        prefix = "policy_final_after_" if is_last else f"policy_after_"
        return os.path.join(
            self.q_matrices_dir, 
            f"{prefix}{episode}episodes.csv"
        )

    def ql_frequency_matrix_file(self, n_episodes, delta, alpha) -> str:
        """ Save the frequency of updating a state matrix as a CSV file"""
        return os.path.join(
            self.q_matrices_dir, 
            f"frequency_after_{n_episodes}_episodes_delta_{delta}_alpha_{alpha}.csv"
        )
    
    @property
    def ql_convergence_plot_file(self) -> str:
        """ Save the convergence plot of Q-Learning training as a PNG file"""
        return os.path.join(
            self.q_matrices_dir, 
            "convergence_q_learning_train.png"
        )

    @property
    def ql_training_metrics_file(self) -> str:
        """ Save the training metrics of the average path length as a CSV file"""
        return os.path.join(
            self.q_matrices_dir,
            "Data_len_path_training.dat"
        )

    # ---------------------------------------- Deep Q-Learning  ------------------------------------------------------------------------------
    @property
    def dqn_checkpoints_dir(self) -> str:
        """Directory to save DQN checkpoints."""
        return os.path.join(
            self.root, 
            "checkpoints"
        )

    def dqn_checkpoint_file(self, epoch: Union[int, str], net_type: str) -> str:
        """ Save the model checkpoints of DQN training as a PTH file
        =======
        Args:
        =======
        @net_type (str) : should be either 'eval' or 'target'
        @epoch (Union[int, str]) : can be an integer or a string to save the "last" models

        Returns:
        -------
            str : path to save the model checkpoint
        """
        prefix = f"epoch_{epoch}" if isinstance(epoch, int) else str(epoch)
        return os.path.join(
            self.dqn_checkpoints_dir, 
            f"{prefix}_{net_type}"
        )
    
    @property
    def dqn_loss_plot(self) -> str:
        """ Save the convergence plot of DQN training as a PNG file"""
        return os.path.join(
            self.dqn_checkpoints_dir, 
            "convergence_deep_q_learning.png"
        )
    
    @property
    def dqn_path_plot(self) -> str:
        """ Save the convergence plot of DQN training for path length as a PNG file"""
        return os.path.join(
            self.dqn_checkpoints_dir, 
            "convergence_deep_q_learning_len_path.png"
        )
    
    def dqn_metrics_csv(self, name: str) -> str:
        """Returns path for losses.csv, avgLenPath.csv, or epsilons.csv.
        Args:
        name (str): should be either "losses", "avgLenPath", or "epsilons"
        """
        return os.path.join(
            self.checkpoints_dir, 
            f"{name}.csv"
        )
    
    @property
    def complexity_report_file(self) -> str:
        """ Returns path for the complexity report of DQN testing."""
        return os.path.join(
            self.dqn_checkpoints_dir, 
            "complexity_report.txt"
        )
    
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
    