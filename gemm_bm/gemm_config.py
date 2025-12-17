import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple
import importlib

# Ensure shared module is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.base_config import BaseConfig

class GemmConfigs(BaseConfig):
    def __init__(self, arch_str: str = "4096"):
        """
        Initialize the GemmConfigs with architecture string.

        Args:
            arch_str (str): Architecture string to identify the configuration.
        """
        super().__init__(arch_str)
        
        # Set channel and pixel parallelism based on architecture
        self.cp = self._arch_map[arch_str]["cp"]
        self.pp = self._arch_map[arch_str]["pp"]
        self.max_input_size = 4096
        self.max_class = 4096

    def generate_batch_sizes(self) -> List[int]:
        """
        Generate a list of valid batch sizes.

        Returns:
            list[int]: A sorted list of batch sizes.
        """
        class_sizes = set()
        for s in range(512, self.max_class, 64):
            class_sizes.add(s)

        class_sizes = sorted(s for s in class_sizes if s <= self.max_class)
        return class_sizes
    
    def generate_input_sizes(self) -> List[int]:
        """
        Generate a list of valid input sizes (e.g., height or width of images).

        Returns:
            list[int]: A sorted list of spatial input sizes.
        """
        input_sizes = set()
        for s in range(512, self.max_input_size, 16):
            input_sizes.add(s)
        input_sizes.add(4096)

        input_sizes = sorted(s for s in input_sizes if s <= self.max_input_size)
        return input_sizes
    
    def generate_configs(self) -> Dict[str, Tuple]:
        """
        Generate the full configuration dictionary based on batch size,
        channel count, input size, kernel size, etc.
        IMPORTANT: USING CONV2d TO SIMULATE LINEAR -> INPUT IS (B, C, H, W)

        Returns:
            dict[str, tuple]: Dictionary mapping config keys to parameter tuples.
        """
        class_sizes = self.generate_batch_sizes()
        input_sizes = self.generate_input_sizes()

        config_dict = {}
        for num_class in class_sizes:
            for num_input in input_sizes:
                if num_input > num_class:
                    key = f"gemm_{num_input}_{num_class}"
                    config_dict[key] = (num_input, num_class)

        return config_dict
    
    def info(self):
        """
        Generate a string with configuration information.

        Returns:
            str: Configuration information string.
        """
        cfgs = self.generate_configs()
        class_list = sorted(set(class_list for _, class_list in cfgs.values()))
        input_list = sorted(set(input_list for input_list, _ in cfgs.values()))
        print("GEMM Config:")
        print(f"  Batch Sizes: {class_list} -> {len(class_list)} unique sizes")
        summary = input_list if len(input_list) <= 10 else "..."
        print(f"  Input Sizes: {summary} -> {len(input_list)} unique sizes")
        print(f"  Total Configs: {len(cfgs)}")

    def load_model_class(self, model_class = "GEMMModel"):
        """
        Load the GEMM model class from the specified import path.
        Returns:
            type[torch.nn.Module]: The GEMM model class.
        """
        import_path = "gemm_model." + model_class
        module_name, class_name = import_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def check_para(self, data):
        """
        Analyze the proportion of configurations that are multiples of cp,
        pp, both, or neither for channels and input sizes.

        Args:
            data (dict or list): Configuration dictionary or list.

        Returns:
            dict: Statistics of each category.
        """
        cp = self.cp
        pp = self.pp

        total = len(data)
        is_cp = 0
        neither = 0

        for v in data.values() if isinstance(data, dict) else data:
            C = v[0]
            is_cp = (C % cp == 0)
            if is_cp:
                cp += 1
            else:
                neither += 1

        print(f"Total: {total}")
        print(f"Both cp({cp}) & pp({pp}) multiples: {cp} ({cp/total:.2%})")
        print(f"Neither:                          {neither} ({neither/total:.2%})")
        return {
            "both": cp,
            "neither": neither,
            "total": total
        }
            
if __name__ == "__main__":
    # Example usage
    arch_str = "4096"
    gemm_configs = GemmConfigs(arch_str)
    print(gemm_configs)
    gemm_configs.info()
    all_configs = gemm_configs.generate_configs()
    gemm_configs.check_para(all_configs)