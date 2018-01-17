#!/usr/bin/env python
# coding=utf-8


import torch
a = torch.rand(5,3)
b = torch.rand(5,3)
print("a+b:{}".format(a+b))

print("a+b:{}".format(torch.add(a,b)))

result = torch.Tensor(5,3)
torch.add(a, b, out=result)  # 把运算结果存储在resulut上

b.add(a) # 把运算结果覆盖掉b

# Tensor 的部分截取

print(b[:, 1])













