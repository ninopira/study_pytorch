import glob
import os

import cv2
import numpy as np
import torch


class CarDataset(torch.utils.data.Dataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    Return:
        img(画像): (3, H, W)
        mask(アノテーション): (1, H, W)
    """

    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
               'tree', 'signsymbol', 'fence', 'car',
               'pedestrian', 'bicyclist', 'unlabelled']

    def __init__(
            self,
            img_dir_path,
            ano_dir_path,
            classes=['car'],
            predict=False,
            aug=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(img_dir_path)
        self.images_fps = [os.path.join(img_dir_path, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(ano_dir_path, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.aug = aug
        self.preprocessing = preprocessing

        self.predict = predict

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        image_ori = image

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augs
        if self.aug:
            sample = self.aug(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if not self.predict:
            return image, mask
        else:
            return image, image_ori, mask, os.path.basename(self.images_fps[i])

    def __len__(self):
        return len(self.ids)
