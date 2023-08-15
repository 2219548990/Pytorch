import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)

# 注意应将input设置为浮点类型，否则maxpool无法处理
# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)
# # 满足最大池化层要求的输入尺寸
# input = torch.reshape(input, (-1, 1, 5, 5))
# print(input.shape)


# 搭建神经网络
class TestModule(nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output


# 创建神经网络
module = TestModule()
# output = TestModule(input)
# print(output)

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("MaxPoolInput", imgs, step)
    output = module(imgs)
    writer.add_images("MaxPoolOutput", output, step)
    step += 1

writer.close()
























