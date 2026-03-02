from enum import Enum
from typing import Tuple

from config.parameters import Parameters
from experiments.store import Store   
from systemModel.channel import Channel
from systemModel.feedback import Feedback
from dataset.noise_calibration import fit_noise


class NoiseMode(str, Enum):
    """
    Enumeration of available noise configuration strategies.

    Values:
    ------
    ANALYTICAL : Use a closed-form analytical noise model.

    FIT : Estimate (fit) noise parameters from the generated dataset.

    REUSE : Load previously fitted noise parameters from disk.
    """
    ANALYTICAL = "analytical"
    FIT = "fit"
    REUSE = "reuse"

class NoiseFactory:
    """ This class is responsible for constructing the noise model parameters associated with a given experiment configuration. 
    
    The noise parameters has 3 options:
        . analytical noise, 
        . noise fitting, 
        . loading fitted noise.
    """
    def get_noise_params(
        self,
        *,
        parameters: Parameters,
        noise_mode: NoiseMode,
        store : Store,
        feedback: Feedback,
        channel: Channel
    ) -> Tuple[float, float]:
        """
        Return noise parameters according to the selected strategy.
        =======
        Args:
        =======
        @ noise_mode (NoiseMode) : Strategy used to determine noise parameters.
        @ paths (ExperimentPaths): Experiment paths object used for saving/loading fits.
        @ feedback (Feedback) : Feedback object used when fitting noise.
        @ channel (Channel) : Channel object used when fitting noise.
        @ parameters (Parameters) : Global experiment parameters.

        Returns:
        -------
        (mean, std)  (Tuple[float, float]) : Noise parameters used by the Probability model.
        """
        if noise_mode == NoiseMode.ANALYTICAL:
            return (0.0, 0.01)
        
        elif noise_mode == NoiseMode.FIT:
            mean, std = fit_noise(store.paths.root, feedback, channel, parameters) 
            return (float(mean), float(std))
        
        elif noise_mode == NoiseMode.REUSE:
            if not store.noise_exists():
                raise FileNotFoundError(
                    f"NoiseMode.REUSE selected but {store.paths.noise_csv} does not exist. "
                    f"Run once with NoiseMode.FIT."
                )
            return store.load_noise()
        
        else:
            raise ValueError(f"Unsupported noise mode: {noise_mode}")

## Must call it like this:
# noise_factory.get_noise_params(
#     noise_mode=NoiseMode.ANALYTICAL,
#     paths=paths,
#     feedback=feedback,
#     channel=channel,
#     parameters=parameters
# )