import abc
from typing import Dict
import numpy as np

class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict, prev_action: np.ndarray | None = None, is_rtc: bool = False) -> Dict:
        """Infer actions from observations."""

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass
