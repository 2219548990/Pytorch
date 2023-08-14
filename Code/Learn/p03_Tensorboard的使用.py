from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# 使用tensorboard需要先创建一个SummaryWriter类的实例，代码运行产生的文件会保存在logs文件目录下
writer = SummaryWriter("logs")

image_path = "dataset/train/ants_image/0013035.jpg"
img = Image.open(image_path)  # <class 'PIL.JpegImagePlugin.JpegImageFile'>
print(type(img))
img_array = np.array(img)  # PIL.JpegImagePlugin.JpegImageFile -> numpy.ndarray
print(type(img_array))  # numpy.ndarray

# # 显示一张图片
# writer.add_image("test", img_array, 1, dataformats='HWC')
#
# # y = x
# for i in range(100):
#     writer.add_scalar("y = x", i, i)
#
# for i in range(100):
#     writer.add_scalar("y = 2x", 2 * i, i)

writer.close()

# -----------------------------------------------------------------------------------------------------------

class TestData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = "dataset/train"
ant_label_dir = "ants_image"
bee_label_dir = "bees_image"
ants_dataset = TestData(root_dir, ant_label_dir)
bees_dataset = TestData(root_dir, bee_label_dir)

# 显示数据集中每张图片
writer = SummaryWriter("logs")
step = 0
for data in ants_dataset:
    img, label = data

    # img_array = np.array(img)
    trans_tensor = transforms.ToTensor()
    img_tensor = trans_tensor(img)
    writer.add_image("ant_image", img_tensor, step)
    step += 1

step = 0
for data in bees_dataset:
    img, label = data
    trans_tensor = transforms.ToTensor()
    img_tensor = trans_tensor(img)
    writer.add_image("bee_image", img_tensor, step)
    step += 1

writer.close()


























