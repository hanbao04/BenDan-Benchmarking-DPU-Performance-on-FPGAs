import torch
import gc
from pytorch_nndct.apis import Inspector as VaiInspector
from gemm_linear_config import GemmConfigs

cfg = GemmConfigs()
configs = cfg.generate_configs()

class VAI_Inspector:
    """
    Assume weight mat is a squared matrix
    """
    def __init__(self, model_class, input_size, class_size):
        self.model_class = model_class
        self.class_size = class_size
        self.input_size = input_size
        self.model = self.model_class(input_size, class_size)

    def inspect(self, name):
        inspector = VaiInspector(cfg.target)
        dummy_input = torch.randn([1, self.input_size]).to(cfg.device)
        inspector.inspect(
            self.model,
            (dummy_input,),
            device=cfg.device,
            output_dir=f"./inspector_output/{name}"
        )

        print(f"\033[44;97mInspector finished for {name}\033[0m")

# Loop over configurations
model_class = cfg.load_model_class()
for i, (name, (input, class_size)) in enumerate(configs.items()):
    # Create inspector and run
    inspector = VAI_Inspector(model_class, input, class_size)
    inspector.inspect(name)
    del inspector
    gc.collect()
    print(f"-------------> Finished {name} (memory cleared)\n")