# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class GEMMModel(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(GEMMModel, self).__init__()
        self.module_0 = py_nndct.nn.Input() #GEMMModel::input_0(GEMMModel::nndct_input_0)
        self.module_1 = py_nndct.nn.Linear(in_features=4096, out_features=4032, bias=False) #GEMMModel::GEMMModel/Linear[gemm]/ret(GEMMModel::nndct_dense_1)

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        return output_module_0
