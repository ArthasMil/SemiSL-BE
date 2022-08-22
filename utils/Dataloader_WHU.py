# encoding=UTF-8
# author: Anzhu Yu
# data: 20210316 12:05

from torch.utils.data import Dataset
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import cv2
import random


def data_augmentation(image, label):
    # Data augmentation
    if random.randint(0, 1):
        image = np.fliplr(image)
        label = np.fliplr(label)
    if random.randint(0, 1):
        image = np.flipud(image)
        label = np.flipud(label)

    if random.randint(0, 1):
        angle = random.randint(0, 3) * 90
        if angle != 0:
            M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1.0)
            image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)
            label = cv2.warpAffine(label, M, (label.shape[1], label.shape[0]), flags=cv2.INTER_NEAREST)

    return image, label


class DatasetWHU_WithName(Dataset):
    def __init__(self, images_path, labels_path, data_aug = False):
        self.images_path = images_path
        self.labels_path = labels_path
        self.images_path_list = os.listdir(images_path)
        self.labels_path_list = os.listdir(labels_path)
        self.data_aug = data_aug

    def __getitem__(self, index):

        image_path = self.images_path_list[index]
        label_path = self.labels_path_list[index]

        imgTemp = cv2.imdecode(np.fromfile(file=os.path.join(self.images_path, image_path), dtype=np.uint8), cv2.COLOR_BGR2RGB)
        labTemp = cv2.imdecode(np.fromfile(file=os.path.join(self.labels_path, label_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if self.data_aug:
            imgTemp, labTemp = data_augmentation(imgTemp, labTemp)
        if np.max(labTemp) == 1:
            labTemp[labTemp == 1] = 255
        image = Image.fromarray(imgTemp )
        label = Image.fromarray(labTemp )
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        transform_lab = transforms.Compose([
            transforms.ToTensor()
        ])

        image = transform(image)
        label = transform_lab(label)
        return image, label, image_path

    def __len__(self):
        return len(self.images_path_list)
