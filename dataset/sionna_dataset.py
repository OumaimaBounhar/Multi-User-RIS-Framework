import pickle
import numpy as np
from typing import Any, List, Tuple


class SionnaRLDataset:
    """
    Minimal adapter to make a Sionna-generated RL dataset compatible
    with the existing DQN pipeline.

    Expected pickle payload format:
    {
        "List_Representant_Classes": ...,
        "n_representant_class": ...,
        ...
    }
    """

    def __init__(self, payload: dict, parameters: Any = None, codebooks: Any = None):
        if "List_Representant_Classes" not in payload:
            raise KeyError(
                "Sionna dataset payload must contain 'List_Representant_Classes'."
            )

        self.payload = payload
        self.List_Representant_Classes = payload["List_Representant_Classes"]
        self.n_representant_class = payload.get("n_representant_class", None)

        # Optional, only to mimic Dataset_probability API
        self.parameters = parameters
        self.codebooks = codebooks

    @classmethod
    def from_pickle(cls, pickle_path: str, parameters: Any = None, codebooks: Any = None):
        with open(pickle_path, "rb") as f:
            payload = pickle.load(f)
        return cls(payload=payload, parameters=parameters, codebooks=codebooks)

    def get_Data(self) -> List[Tuple[np.ndarray, List[np.ndarray]]]:
        return self.List_Representant_Classes

    def get_params_codebook(self):
        return self.parameters, self.codebooks

    def get_class_counts(self):
        return self.n_representant_class