import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class TestModule(nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        # # 输入：3 @ 32 × 32
        # # 输出：32 @ 32 × 32
        # self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        # # 输入：32 @ 32 × 32
        # # 输出：32 @ 16 × 16
        # self.maxpool1 = MaxPool2d(2)
        # # 输入：32 @ 16 × 16
        # # 输出：32 @ 16 × 16
        # self.conv2 = Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        # # 输入：32 @ 16 × 16
        # # 输出：32 @ 8 × 8
        # self.maxpool2 = MaxPool2d(2)
        # # 输入：32 @ 8 × 8
        # # 输出：64 @ 8 × 8
        # self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        # # 输入：64 @ 8 × 8
        # # 输出：64 @ 4 × 4
        # self.maxpool3 = MaxPool2d(2)
        # # 输入：64 @ 4 × 4
        # # 输出：1024
        # self.flatten = Flatten()
        # # 输入：1024
        # # 输出：64
        # self.linear1 = Linear(1024, 64)
        # # 输入：64
        # # 输出：10
        # self.linear2 = Linear(64, 10)

        self.model1 = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        x = self.model1(x)
        return x


# 实例化网络
module = TestModule()
print(module)
# 验证网络结构是否正确
# torch.ones可以产生任意形状的数据，值全为1
input = torch.ones(64, 3, 32, 32)
output = module(input)
print(output.shape)

# 使用tensorboard可视化模型
writer = SummaryWriter("logs")
writer.add_graph(module, input)
writer.close()

















































