#!/usr/bin/env python
# coding=utf-8

import torch

a = torch.ones(5)
b = a.numpy() # a为tensor
print("a:{}".format(a))
print("b:{}".format(b))


# array -> tensor
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a) # a为numpy的array
print("a:{}".format(a))
print("b:{}".format(b))


# CUDA
print(torch.cuda.is_available())  # 看看是否支持CUDA





