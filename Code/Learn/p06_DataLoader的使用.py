import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备数据集
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
# 加载数据集
'''
DataLoader的作用是加载数据集
    dataset:要加载的数据集
    batch_size:每次取的图片
    shuffle:如果为True，每次抓取过数据后打乱数据，如果为False，则不打乱
    num_workers:加载数据时使用的进程数
    drop_last:按照batch_size抓取数据，如果最后剩余的数据不能被batch_size整除，drop_last等于True则放弃剩余的数据
'''
test_loader = DataLoader(dataset=test_data, batch_size=64,shuffle=True, num_workers=0, drop_last=False)

writer = SummaryWriter("logs")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch: {}".format(epoch), imgs, step)
        step += 1

writer.close()
























