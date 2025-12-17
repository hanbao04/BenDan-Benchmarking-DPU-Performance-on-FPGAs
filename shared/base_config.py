from abc import ABC, abstractmethod
from typing import Dict, Tuple
import torch

class BaseConfig(ABC):
    """
    Abstract base class for hardware-specific layer configuration generation.
    Defines common architecture mapping and enforces implementation of core methods.
    """

    _arch_map = {
        "512":   {"cp": 8,  "pp": 4},
        "800":   {"cp": 10, "pp": 4},
        "1024":  {"cp": 8,  "pp": 8},
        "1152":  {"cp": 12, "pp": 4},
        "1600":  {"cp": 10, "pp": 8},
        "2304":  {"cp": 12, "pp": 8},
        "3136":  {"cp": 14, "pp": 8},
        "4096":  {"cp": 16, "pp": 8},
    }

    def __init__(self, arch_name="4096"):
        """
        Initialize the base configuration.

        Args:
            arch_name (str): Target DPU architecture name (e.g., "4096").

        Raises:
            ValueError: If the provided architecture name is not recognized.
        """
        self.device = torch.device("cpu")
        self.target = f"DPUCZDX8G_ISA1_B{arch_name}"

        arch_str = str(arch_name)
        if arch_str not in self._arch_map:
            raise ValueError(f"Unknown arch_name '{arch_name}'. Valid options: {list(self._arch_map.keys())}")

        self.cp = self._arch_map[arch_str]["cp"]  # channel parallelism
        self.pp = self._arch_map[arch_str]["pp"]  # pixel parallelism

    @abstractmethod
    def generate_configs(self) -> Dict[str, Tuple]:
        """
        Generate the full configuration dictionary based on batch size,
        channel count, input size, kernel size, etc.

        Returns:
            dict[str, tuple]: Dictionary mapping config keys to parameter tuples.
        """
        pass

    @abstractmethod
    def load_model_class(model_class):
        """
        Load a model class to generate configurations.

        Args:
            model_class (type): The model class to load.

        Returns:
            BaseConfig: An instance of the configuration class.
        """
        pass

    @abstractmethod
    def info(self) -> None:
        """
        Print summary statistics of the generated configurations.
        """
        pass
    
    @abstractmethod
    def load_model_class(self, model_class):

        """
        Load the model class for generating configurations.

        Args:
            model_class (str): The name of the model class to load.

        Returns:
            type[torch.nn.Module]: The loaded model class.
        """
        pass
    
    def __str__(self):
        return f" ------> HW configuration with arch_name = {self.target}, cp = {self.cp}, pp = {self.pp} <------"
    