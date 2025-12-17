# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class CONVModel(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(CONVModel, self).__init__()
        self.module_0 = py_nndct.nn.Input() #CONVModel::input_0(CONVModel::nndct_input_0)
        self.module_1 = py_nndct.nn.Conv2d(in_channels=154, out_channels=154, kernel_size=[7, 7], stride=[1, 1], padding=[3, 3], dilation=[1, 1], groups=1, bias=False) #CONVModel::CONVModel/Conv2d[conv]/ret(CONVModel::nndct_conv2d_1)

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        return output_module_0
