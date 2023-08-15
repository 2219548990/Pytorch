import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader


dataset = torchvision.datasets.CIFAR10("./dataset", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)


class TestModule(nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        # 将线性层的输入设置为196008，输出设置为10
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output


module = TestModule()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    # 满足线性层的输入形状
    # output = torch.reshape(imgs, (1, 1, 1, -1))
    output = torch.flatten(imgs)  # 可以直接将数据展开成一行
    print(output.shape)
    output = module(output)
    print(output.shape)































