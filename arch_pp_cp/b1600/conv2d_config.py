import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple
import importlib

# Ensure shared module is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from shared.base_config import BaseConfig

class ConvConfigs(BaseConfig):
    def __init__(self, arch_str: str):
        """
        Initialize ConvConfigs with the architecture string.

        Args:
            arch_str (str): Architecture identifier.
        """
        super().__init__(arch_str)
        # Set channel and pixel parallelism based on architecture
        self.cp = self._arch_map[arch_str]["cp"]
        self.pp = self._arch_map[arch_str]["pp"]
        self.channel_size = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.max_batch_size = 1 
        self.kernel_size = [3, 5, 7]                # Valid kernel sizes
        self.img_size = [32, 64, 128, 256, 512]     # Aligned input sizes

    def generate_batch_sizes(self) -> List[int]:
        """
        Generate a list of valid batch sizes.

        Returns:
            list[int]: Sorted list of batch sizes.
        """
        return [self.max_batch_size]
    
    def generate_input_sizes(self) -> List[int]:
        """
        Generate a list of valid input sizes (height/width of images), 
        including both aligned and misaligned sizes.

        Returns:
            list[int]: Sorted list of input sizes.
        """

        return self.img_size

    def generate_channel_sizes(self) -> List[int]:
        """
        Generate a list of valid channel sizes for convolution operations, 
        including both aligned and misaligned cases.

        Returns:
            list[int]: Sorted list of channel sizes.
        """

        return self.channel_size
    
    def generate_kernel_size(self) -> List[int]:
        """
        Generate a list of valid kernel sizes for convolution operations.

        Returns:
            list[int]: List of valid kernel sizes.
        """
        return self.kernel_size

    def generate_configs(self) -> Dict[str, Tuple]:
        """
        Generate a dictionary of all possible configuration tuples 
        (B, C, I, K), given batch size, channel size, input size, and kernel size.

        Returns:
            dict[str, tuple]: Keyed by config name, value is (B, C, I, K).
        """
        batch_sizes = self.generate_batch_sizes()
        channels_list = self.generate_channel_sizes()
        img_sizes = self.generate_input_sizes()
        kernel_sizes = self.generate_kernel_size()
        config_dict = {}
        for B in batch_sizes:
            for C in channels_list:
                for I in img_sizes:
                    for K in kernel_sizes:
                        key = f"conv_{B}_{C}_{I}_{K}"
                        config_dict[key] = (B, C, I, K)
        return config_dict
    
    def info(self) -> None:
        """
        Print a summary of the configuration space.
        """
        cfgs = self.generate_configs()
        batch_list = sorted(set(cfg[0] for cfg in cfgs.values()))
        ch_list = sorted(set(cfg[1] for cfg in cfgs.values()))
        img_list = sorted(set(cfg[2] for cfg in cfgs.values()))
        k_list = sorted(set(cfg[3] for cfg in cfgs.values()))

        print("CONV Config:")
        print(f"    Batch sizes: {len(batch_list)} -> {batch_list}")
        print(f"    Channels: {len(ch_list)} -> {ch_list}")
        if (len(img_list) > 10): 
            summary_img = "..."
            print(f"    Input sizes: {len(img_list)} -> {summary_img}")
        else:
            print(f"    Input sizes: {len(img_list)} -> {img_list}")
        print(f"    Kernel sizes: {len(k_list)} -> {k_list}")
        print("Total number of configurations:", len(cfgs))

    def load_model_class(self, model_class = "CONVModel"):
        """
        Dynamically load the CONV model class from specified import path.

        Args:
            model_class (str): Name of the model class.

        Returns:
            type[torch.nn.Module]: The model class.
        """
        import_path = "conv2d_model." + model_class
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
        cp_and_pp = 0
        cp_only = 0
        pp_only = 0
        neither = 0

        for v in data.values() if isinstance(data, dict) else data:
            C = v[1]
            I = v[2]
            is_cp = (C % cp == 0)
            is_pp = (I % pp == 0)
            if is_cp and is_pp:
                cp_and_pp += 1
            elif is_cp:
                cp_only += 1
            elif is_pp:
                pp_only += 1
            else:
                neither += 1

        print(f"Total: {total}")
        print(f"Both cp({cp}) & pp({pp}) multiples: {cp_and_pp} ({cp_and_pp/total:.2%})")
        print(f"Only cp({cp}) multiples:           {cp_only} ({cp_only/total:.2%})")
        print(f"Only pp({pp}) multiples:           {pp_only} ({pp_only/total:.2%})")
        print(f"Neither:                          {neither} ({neither/total:.2%})")
        return {
            "both": cp_and_pp,
            "cp_only": cp_only,
            "pp_only": pp_only,
            "neither": neither,
            "total": total
        }
        
if __name__ == "__main__":
    # Example usage
    arch_str = "1600"
    conv_configs = ConvConfigs(arch_str)
    all_configs = conv_configs.generate_configs()

    print(conv_configs)
    conv_configs.info()
    conv_configs.check_para(all_configs)
