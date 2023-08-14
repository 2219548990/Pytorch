import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)

# 查看数据集中第一张图片
print(test_set[0])  # torch.Tensor
# 查看数据集的classes属性，classes是存储图片类别的list
print(test_set.classes)
# 获取数据集中的数据
img, target = test_set[0]
print(img)
print(target)
print(test_set.classes[target])

writer = SummaryWriter("logs")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()





























