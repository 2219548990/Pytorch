from torch.utils.data import Dataset
from PIL import Image  # python自带的用于获取图片的工具
import os

'''
    使用Dataset必须先导包：from torch.utils.data import Dataset
    Dataset是一个抽象类，每一个继承它的子类必须重写它的“__getitem__”方法

'''


class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir  # 数据集根目录地址
        self.label_dir = label_dir  # 每个数据类文件夹名称
        self.path = os.path.join(self.root_dir, label_dir)  # 拼接两个地址
        self.img_path = os.listdir(self.path)  # 将self.path路径下所有图片名称做成一个list

    # 继承Dataset类的子类必须重写这个方法
    def __getitem__(self, idx):
        img_name = self.img_path[idx]  # 获取图片名称
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)  # 获取图片完整路径
        img = Image.open(img_item_path)  # 打开图片
        label = self.label_dir  # 获取图片标签
        return img, label  # 返回图片和标签

    # 继承Dataset类的子类可选择重写这个方法
    def __len__(self):
        return len(self.img_path)


root_dir = "dataset/train"
ant_label_dir = "ants_image"
bee_label_dir = "bees_image"

ants_dataset = MyData(root_dir, ant_label_dir)
bees_dataset = MyData(root_dir, bee_label_dir)
print(len(ants_dataset))

img, label = ants_dataset[0]
img.show()

train_dataset = ants_dataset + bees_dataset
print(len(train_dataset))



































