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
    
    # --------------------------- Parameters ---------------------------
    @property # allows to use a method like an attribute
    def config_dir(self) -> str:
        return os.path.join(self.root, "config")
    
    def params_file(self, mode: str) -> str:
        return os.path.join(self.config_dir, f"params_{mode}.txt")

    def params_resume_training_file(self, mode: str) -> str:
        return os.path.join(self.config_dir, f"params_resume_training_{mode}.txt")

    # --------------------------- Dataset ---------------------------
    @property
    def dataset_dir(self) -> str:
        return os.path.join(self.root, "dataset")

    @property
    def dataset_pickle(self) -> str:
        return os.path.join(self.dataset_dir, "Dataset.pickle")
    
    # --------------------------- Q-Learning  ---------------------------
    @property
    def q_learning_dir(self) -> str:
        return os.path.join(self.root, "q_learning")
    
    @property
    def q_matrices_dir(self) -> str:
        return os.path.join(
            self.q_learning_dir, 
            "Q_matrices"
        )

    @property
    def q_policies_dir(self) -> str:
        return os.path.join(self.q_learning_dir, "policies")

    @property
    def q_plots_dir(self) -> str:
        return os.path.join(self.q_learning_dir, "plots")
    
    @property
    def q_metrics_dir(self) -> str:
        return os.path.join(self.q_learning_dir, "metrics")

    def q_matrix_file(self, episode: int) -> str:
        """ Save the Q-matrix as a CSV file
        Args:
            episode (int): The episode number after which the Q-matrix is saved
        Returns:
            str: The path to save the Q-matrix CSV file
        """
        
        return os.path.join(
            self.q_matrices_dir, 
            f"Q_matrix_after_{episode}episodes.csv"
        )
    
    def ql_frequency_matrix_file(self, n_episodes, delta, alpha) -> str:
        """ Save the frequency of updating a state matrix as a CSV file"""
        return os.path.join(
            self.q_matrices_dir, 
            f"frequency_after_{n_episodes}_episodes_delta_{delta}_alpha_{alpha}.csv"
        )

    def policy_matrix_file(self, episode: int) -> str:
        """ Save the policy matrix as a CSV file
        Args:
            episode (int): The episode number after which the policy matrix is saved
        Returns:
            str: The path to save the policy matrix CSV file.
        """
        
        return os.path.join(
            self.q_policies_dir, 
            f"policy_after_{episode}episodes.csv"
        )

    @property
    def ql_convergence_plot_file(self) -> str:
        """ Save the convergence plot of Q-Learning training as a PNG file"""
        return os.path.join(
            self.q_plots_dir, 
            "convergence_q_learning_train.png"
        )

    @property
    def ql_training_metrics_file(self) -> str:
        """ Save the training metrics of the average path length as a CSV file"""
        return os.path.join(
            self.q_metrics_dir,
            "Data_len_path_training.dat"
        )

    # --------------------------- Deep Q-Learning  ---------------------------
    @property
    def dqn_dir(self) -> str:
        return os.path.join(self.root, "deep_q_learning")
    
    @property
    def dqn_checkpoints_dir(self) -> str:
        """Directory to save DQN checkpoints."""
        return os.path.join(
            self.dqn_dir, 
            "checkpoints"
        )
    
    @property
    def dqn_metrics_dir(self) -> str:
        return os.path.join(self.dqn_dir, "metrics")

    @property
    def dqn_plots_dir(self) -> str:
        return os.path.join(self.dqn_dir, "plots")

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
            f"{prefix}_{net_type}.pth"
        )
    
    @property
    def dqn_loss_plot(self) -> str:
        """ Save the convergence plot of DQN training as a PNG file"""
        return os.path.join(
            self.dqn_plots_dir, 
            "convergence_deep_q_learning_loss.png"
        )
    
    @property
    def dqn_path_plot(self) -> str:
        """ Save the convergence plot of DQN training for path length as a PNG file"""
        return os.path.join(
            self.dqn_plots_dir, 
            "convergence_deep_q_learning_len_path.png"
        )
    
    @property
    def dqn_reward_plot(self) -> str:
        return os.path.join(
            self.dqn_plots_dir,
            "convergence_deep_q_learning_reward.png"
        )
    
    @property
    def dqn_grad_norm_plot(self) -> str:
        return os.path.join(
            self.dqn_plots_dir,
            "convergence_deep_q_learning_grad_norm.png"
        )

    @property
    def dqn_grad_norm_max_plot(self) -> str:
        return os.path.join(
            self.dqn_plots_dir,
            "convergence_deep_q_learning_grad_norm_max.png"
        )

    def dqn_metrics_csv(self, name: str) -> str:
        """Returns path for losses.csv, avgLenPath.csv, or epsilons.csv.
        Args:
        name (str): should be either "losses", "avgLenPath", or "epsilons"
        """
        return os.path.join(
            self.dqn_metrics_dir, 
            f"{name}.csv"
        )

    @property
    def dqn_training_state_file(self) -> str:
        return os.path.join(self.dqn_checkpoints_dir, "training_state.pth")
    
    @property
    def complexity_report_file(self) -> str:
        """ Returns path for the complexity report of DQN testing."""
        return os.path.join(
            self.dqn_checkpoints_dir, 
            "complexity_report.txt"
        )

    # --------------------------- Test sub-folders ---------------------------
    @property
    def tests_dir(self) -> str:
        return os.path.join(self.root, "tests")
    
    @property
    def test_probabilities_dir(self) -> str:
        return os.path.join(self.tests_dir, "probabilities")

    @property
    def test_strengths_dir(self) -> str:
        return os.path.join(self.tests_dir, "strengths")

    @property
    def test_summaries_dir(self) -> str:
        return os.path.join(self.tests_dir, "summaries")

    @property
    def test_plots_dir(self) -> str:
        return os.path.join(self.tests_dir, "plots")

    @property
    def test_probability_plots_dir(self) -> str:
        return os.path.join(self.test_plots_dir, "probability")

    @property
    def test_strength_plots_dir(self) -> str:
        return os.path.join(self.test_plots_dir, "strength")

    @property
    def test_success_plots_dir(self) -> str:
        return os.path.join(self.test_plots_dir, "success")
    
    # --------------------------- Test files ---------------------------
    def test_summary_file(self, epoch: int, snr: Union[int, float]) -> str:
        return os.path.join(
            self.test_summaries_dir,
            f"data_epoch_{epoch}_snr_{snr}.dat"
        )

    def test_probability_file(self, epoch: int, snr: Union[int, float]) -> str:
        return os.path.join(
            self.test_probabilities_dir,
            f"probability_epoch_{epoch}_snr_{snr}.dat"
        )

    def test_strength_file(self, epoch: int, snr: Union[int, float]) -> str:
        return os.path.join(
            self.test_strengths_dir,
            f"strength_epoch_{epoch}_snr_{snr}.dat"
        )

    def test_probability_plot(self, epoch: int, snr: Union[int, float]) -> str:
        return os.path.join(
            self.test_probability_plots_dir,
            f"Image_snr_{snr}_epoch_{epoch}.png"
        )

    def test_strength_plot(self, epoch: int, snr: Union[int, float]) -> str:
        return os.path.join(
            self.test_strength_plots_dir,
            f"Image_strength_epoch_{epoch}_snr_{snr}.png"
        )

    def test_success_plot(self, epoch: int, snr: Union[int, float]) -> str:
        return os.path.join(
            self.test_success_plots_dir,
            f"successful_episodes_epoch_{epoch}_snr_{snr}.png"
        )
    # --------------------------- Noise files ---------------------------
    @property
    def noise_dir(self) -> str:
        return os.path.join(self.root, "noise")
    
    @property
    def noise_csv(self) -> str:
        return os.path.join(self.noise_dir, "Noise_parameters.csv")
    
    def noise_csv_per_snr(self, snr: Union[int, float]) -> str:
        return os.path.join(
            self.noise_dir,
            f"Noise_parameters_snr_{snr}.csv"
        )

    @property
    def noise_csv_all(self) -> str:
        return os.path.join(self.noise_dir, "Noise_parameters_all.csv")

    def noise_histogram_plot(self, snr: Union[int, float]) -> str:
        return os.path.join(
            self.noise_dir,
            f"Noise_histogram_snr_{snr}.png"
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

        dirs_to_create = [
            self.paths.root,
            self.paths.config_dir,
            self.paths.dataset_dir,
            self.paths.noise_dir,

            self.paths.dqn_dir,
            self.paths.dqn_checkpoints_dir,
            self.paths.dqn_metrics_dir,
            self.paths.dqn_plots_dir,

            self.paths.q_learning_dir,
            self.paths.q_matrices_dir,
            self.paths.q_policies_dir,
            self.paths.q_metrics_dir,
            self.paths.q_plots_dir,

            self.paths.tests_dir,
            self.paths.test_probabilities_dir,
            self.paths.test_strengths_dir,
            self.paths.test_summaries_dir,
            self.paths.test_plots_dir,
            self.paths.test_probability_plots_dir,
            self.paths.test_strength_plots_dir,
            self.paths.test_success_plots_dir,
        ]

        for d in dirs_to_create:
            os.makedirs(d, exist_ok=True)

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
    
