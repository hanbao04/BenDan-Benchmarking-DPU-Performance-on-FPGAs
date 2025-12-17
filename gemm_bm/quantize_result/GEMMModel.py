# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class GEMMModel(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(GEMMModel, self).__init__()
        self.module_0 = py_nndct.nn.Input() #GEMMModel::input_0(GEMMModel::nndct_input_0)
        self.module_1 = py_nndct.nn.Conv2d(in_channels=4096, out_channels=4032, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #GEMMModel::GEMMModel/Conv2d[gemm]/ret(GEMMModel::nndct_conv2d_1)

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        return output_module_0
