import torch
import gc
from pytorch_nndct.apis import Inspector as VaiInspector
from conv2d_config import ConvConfigs

cfg = ConvConfigs("1600")
configs = cfg.generate_configs()

class VAI_Inspector:
    """
    Assume weight mat is a squared matrix
    """
    def __init__(self, model_class, batch_sizes, channel_size, img_sizes, kernel_size):
        self.model_class = model_class
        self.channel_size = channel_size
        self.kernel_size = kernel_size
        self.batch_size = batch_sizes
        self.img_sizes = img_sizes
        self.model = self.model_class(channel_size, kernel_size)

    def inspect(self, name):
        inspector = VaiInspector(cfg.target)
        dummy_input = torch.randn([self.batch_size, self.channel_size, self.img_sizes, self.img_sizes]).to(cfg.device)
        inspector.inspect(
            self.model,
            (dummy_input,),
            device=cfg.device,
            output_dir=f"./inspector_output/{name}"
        )

        print(f"\033[44;97mInspector finished for {name}\033[0m")

model_class = cfg.load_model_class("CONVModel")
# Loop over configurations
for i, (name, (batch, channel, img, kernel)) in enumerate(configs.items()):
    # Create inspector and run
    inspector = VAI_Inspector(model_class, batch, channel, img, kernel)
    inspector.inspect(name)
    del inspector
    gc.collect()
    print(f"-------------> Finished {name} (memory cleared)\n")