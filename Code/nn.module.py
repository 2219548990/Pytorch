import torch
from torch import nn


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


module = MyModule()
x = torch.tensor(1)
output = module(x)
print(output)




































