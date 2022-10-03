import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import os
from PIL import Image, ImageFile
import numpy as np


class MyDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, index):
        return super().__getitem__(index)

    def __len__(self) -> int:
        return super().__len__()


class PennFudanDataset(Dataset):
    def __init__(self, root, transforms, ) -> None:
        super().__init__()

        self.root = root
        self.transform = transforms
        self.images = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.masks[index]

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path)

        # 把PIL图像转换为numpy数组
        mask = np.array(mask)
        # 得到该图像总共是什么类别的物体
        obj_ids = np.unique(mask)
        # 第一个为背景，所以去掉
        obj_ids = obj_ids[1:]
        
        masks = mask == obj_ids[:, None, None]
        
        
        

    def __len__(self) -> int:
        return super().__len__()
