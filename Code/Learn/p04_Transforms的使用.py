from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
import cv2

# transforms其实是一个py文件，里面内置很多类，可以作为一个工具箱使用

img_path = "dataset/train/ants_image/7759525_1363d24e88.jpg"
img = cv2.imread(img_path)
print(type(img))  # numpy.ndarray

# 创建一个ToTensor类的实例
trans_tensor = transforms.ToTensor()
# 将图片类型转换为Tensor
img_tensor = trans_tensor(img)
print(type(img_tensor))  # torch.Tensor

# ------------------------------------------------------------------------------------
writer = SummaryWriter("logs")
img = Image.open("dataset/image/R-C.jpg")
print(np.shape(img))

# ToTensor
trans_tensor = transforms.ToTensor()
img_tensor = trans_tensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([3, 2, 1], [2, 4, 6])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm)

# Resize
print(img.size)
trans_resize = transforms.Resize((400, 400))
img_resize = trans_resize(img)
print(img_resize.size)
img_resize = trans_tensor(img_resize)
writer.add_image("Resize", img_resize)

# Compose
trans_compose = transforms.Compose([transforms.Resize(512), trans_tensor])
img_compose = trans_compose(img)
writer.add_image("Compose", img_compose)

# RandomCrop
trans_random = transforms.RandomCrop((100, 200))
trans_compose_2 = transforms.Compose([trans_random, trans_tensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)
    print(np.shape(img_crop))






writer.close()
























