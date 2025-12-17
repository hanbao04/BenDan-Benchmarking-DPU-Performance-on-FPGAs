import torch
import torch.nn as nn
from gemm_config import GemmConfigs

cfg = GemmConfigs("4096")
configs = cfg.generate_configs()

# Define GEMM model using nn.Linear (which is equivalent to GEMM)
class GEMMModel(nn.Module):
    def __init__(self, input_size, class_size):
        super(GEMMModel, self).__init__()
        self.gemm = nn.Conv2d(
            in_channels=input_size, 
            out_channels=class_size,
            kernel_size=1,
            bias=False
        )
        # layers = []
        # for _ in range(100):
        #     layers.append(nn.Conv2d(input_size, input_size, kernel_size=1, bias=False))
        #     layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        # self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.gemm(x)
        
        # return self.model(x)

# Loop over configurations
for i, (name, (input, num_class)) in enumerate(configs.items()):
    print(f"[{i:03}] Generating weights for {name}")

    # Instantiate model
    model = GEMMModel(input, num_class).to(cfg.device)

    # Dummy input for weight generation
    dummy_input = torch.randn([1, input, 1, 1]).to(cfg.device) 

    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)

    # Save model weights
    torch.save(model.state_dict(), f"weight_output/{name}.pt")
    print(f"-------------> Saved: weight_output/{name}.pt\n")
