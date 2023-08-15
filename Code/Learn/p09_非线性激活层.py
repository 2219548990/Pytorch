import torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备数据集
dataset = torchvision.datasets.CIFAR10("./dataset", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)


# 搭建神经网络
class TestModule(nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, input):
        output = self.sigmoid(input)
        return output


# 实例化神经网络
module = TestModule()

# 显示数据到Tensorboard中
writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("SigmoidInput", imgs, global_step=step)
    output = module(imgs)
    writer.add_images("SigmoidOutput", output, global_step=step)
    step += 1

writer.close()






















