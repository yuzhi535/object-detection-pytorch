from importlib.resources import read_text
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import os
import cv2 as cv
import numpy as np
import xml.etree.ElementTree as ET


class MyDataset(Dataset):
    def __init__(self, imgs_path, xmls_path) -> None:
        super().__init__()
        object_names = {'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'}
        self.object_ids = {name: i for i, name in enumerate(object_names)}
        self.object_names = {id: name for name, id in self.object_ids.items()}
        self.images = os.listdir(imgs_path)
        self.images = sorted(self.images, key=lambda x: int(x.split('.')[0]))
        self.images = [os.path.join(imgs_path, image) for image in self.images]
        self.xmls = os.listdir(xmls_path)
        self.xmls = sorted(self.xmls, key=lambda x: int(x.split('.')[0]))
        self.xmls = [os.path.join(xmls_path, xml) for xml in self.xmls]

    def __getitem__(self, index):
        image_path = self.images[index]
        xml_path = self.xmls[index]

        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        gt_boxes = self.read_xml(xml_path)

        aug_image, gt_boxes = self.augmentate(image, gt_boxes)

        return {'image': aug_image, 'label': gt_boxes, 'origin_image': image}

    def __len__(self) -> int:
        return len(self.images)

    def read_xml(self, xml_path):
        object_list = []

        tree = ET.parse(open(xml_path, 'r'))
        root = tree.getroot()

        objects = root.findall("object")
        for _object in objects:  # loop through multiple objects
            bndbox = _object.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            class_name = _object.find('name').text
            object_list.append({'x1': xmin, 'x2': xmax, 'y1': ymin,
                                'y2': ymax, 'class': self.object_ids[class_name]})

        return object_list

    # 数据增强
    def augmentate(self, image, gt_boxes):
        augmentations = A.Compose([
            A.RandomSizedBBoxSafeCrop(
                300, 300, erosion_rate=0.0, interpolation=1, always_apply=False, p=0.8),
            A.HorizontalFlip(p=1),
            A.RandomBrightnessContrast(p=0.2),
        ], bbox_params=A.BboxParams(format='pascal_voc'))

        original_bboxes = [
            [bbox['x1'],
             bbox['y1'],
             bbox['x2'],
             bbox['y2'],
             self.object_names.get(bbox['class'])]
            for bbox in gt_boxes]

        aug = augmentations(image=image, bboxes=original_bboxes)
        image, bboxes = aug['image'], aug['bboxes']
        gt_boxes = [{'x1': bbox[0], 'y1': bbox[1], 'x2': bbox[2], 'y2': bbox[3],
                     'class': self.object_ids[bbox[4]]} for idx, bbox in enumerate(bboxes)]
        return image, gt_boxes
