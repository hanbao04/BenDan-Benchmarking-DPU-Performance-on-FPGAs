import torch
import torch.nn as nn
from gemm_linear_config import GemmConfigs

cfg = GemmConfigs("4096")
configs = cfg.generate_configs()

# Define GEMM model using nn.Linear (which is equivalent to GEMM)
class GEMMModel(nn.Module):
    def __init__(self, input_size, class_size):
        super(GEMMModel, self).__init__()
        self.gemm = nn.Linear(
            in_features=input_size,
            out_features=class_size,
            bias=False
        )

    def forward(self, x):
        return self.gemm(x)

# Loop over configurations
for i, (name, (input, class_size)) in enumerate(configs.items()):
    print(f"[{i:03}] Generating weights for {name}")

    # Instantiate model
    model = GEMMModel(input, class_size).to(cfg.device)

    # Dummy input for weight generation
    dummy_input = torch.randn([1, input]).to(cfg.device) 

    # Forward pass
    model.eval()
    with torch.no_grad():
        # Forward pass to initialize weights
        _ = dummy_input.to(cfg.device)

    # Save model weights
    torch.save(model.state_dict(), f"weight_output/{name}.pt")
    print(f"-------------> Saved: weight_output/{name}.pt\n")
