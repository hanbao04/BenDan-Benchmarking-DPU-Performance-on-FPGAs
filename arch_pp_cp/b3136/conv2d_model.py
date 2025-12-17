import torch
import torch.nn as nn
from conv2d_config import ConvConfigs
import math

cfg = ConvConfigs("3136")
configs = cfg.generate_configs()

class CONVModel(nn.Module):
    def __init__(self, channel_size, kernel_size):
        super(CONVModel, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=channel_size, 
            out_channels=channel_size,  
            kernel_size=kernel_size,
            # padding=(kernel_size-1) // 2,  # Assuming same padding
            padding=math.ceil((kernel_size-1) / 2),
            bias=False
        )

    def forward(self, x):
        return self.conv(x)


# Loop over configurations
for i, (name, (batch, channel, img, kernel)) in enumerate(configs.items()):
    print(f"[{i:03}] Generating weights for {name}")

    # Instantiate model
    model = CONVModel(channel, kernel).to(cfg.device)

    # Dummy input for weight generation
    # assume squared input image
    dummy_input = torch.randn([batch, channel, img, img]).to(cfg.device) 

    # Forward pass
    model.eval()
    with torch.no_grad():
        # Forward pass to initialize weights
        _ = model(dummy_input).to(cfg.device)

    # Save model weights
    torch.save(model.state_dict(), f"weight_output/{name}.pt")
    print(f"-------------> Saved: weight_output/{name}.pt\n")
