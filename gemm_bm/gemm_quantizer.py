import torch
from tqdm import tqdm
import gc
from pytorch_nndct.apis import torch_quantizer
from gemm_config import GemmConfigs

cfg = GemmConfigs()
configs = cfg.generate_configs()

class VAI_Quantizer:
    def __init__(self, model_class, weight_path, input_size, class_size):
        self.model_class = model_class
        self.weight_path = weight_path
        self.input_size = input_size
        self.class_size = class_size
        self.model = self.model_class(input_size, class_size)
        self.quant_model = None

    def quantizer_calib(self):
        device = cfg.device
        if self.weight_path is not None:
            self.model.load_state_dict(torch.load(self.weight_path, map_location=device))

        input_data = torch.randn([1, self.input_size, 1, 1]).to(device)

        quantizer = torch_quantizer('calib', self.model, (input_data,), device=device)
        self.quant_model = quantizer.quant_model

        self.model.eval()

        # Perform calibration using random data
        random_input = torch.randn([1, self.input_size, 1, 1]).to(device)
        _ = self.quant_model(random_input)

        quantizer.export_quant_config()
        print("Calibration config exported successfully.")

    def quantizer_export(self, name, onnx_export=False):
        input = torch.randn([1, self.input_size, 1, 1])
        quantizer = torch_quantizer('test', self.model, (input), device=cfg.device)
        quant_model = quantizer.quant_model

        _ = quant_model(input)
        
        quantizer.export_xmodel(deploy_check=False, output_dir=f"./quantizer_output/{name}")

        if onnx_export:
            quantizer.export_onnx_model(output_dir=f"./quantizer_output/{name}")

        print("Quantized model exported successfully.")

model_class = cfg.load_model_class()
for i, (name, (input, num_class)) in enumerate(configs.items()):
        weight_path = f"weight_output/{name}.pt"

        quant = VAI_Quantizer(model_class, weight_path, input, num_class)
        quant.quantizer_calib()
        quant.quantizer_export(name)
        del quant
        gc.collect()

        print(f"-------------> Finished {name} (memory cleared)\n")