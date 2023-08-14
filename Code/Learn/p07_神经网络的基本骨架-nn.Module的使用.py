import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备数据集
dataset = torchvision.datasets.CIFAR10("./dateset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
# 加载数据集
dataloader = DataLoader(dataset=dataset, batch_size=64)
# 搭建神经网络
class TestModule(nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

module = TestModule()

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    output = module(imgs)
    print(output.shape)
    writer.add_images("input", imgs, step)

    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)
    step += 1

writer.close()

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()






























