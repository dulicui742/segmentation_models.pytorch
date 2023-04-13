import os
import cv2
import random
import pandas as pd
import torch
import numpy as np
import SimpleITK as sitk

from torchvision.io import read_image
from torch.utils.data import Dataset
from PIL import Image


class SegDataset(Dataset):
    def __init__(
        self,
        uid_file, ## "/Data/data10/dst/50_10_增强全部件/train.txt"
        base_path, ## /Data/data10/dst/dicom/
        # model_name,
        stl_names,
        height=512,
        width=512,
        channels=1,
        windowlevel=-600,
        windowwidth=2000,
        transform=None,
        target_transform=None,
        image_folder_name="dicom",
        mask_folder_name="mask",
        status=True, # True for training, False for test
    ):
        ## TO DO: 完成数据列表的统计，transform
        ## input：list_txt, 

        self.base_path = base_path
        # self.model_name = model_name
        self.image_folder_name = image_folder_name
        self.mask_folder_name = mask_folder_name
        self.transform = transform
        self.target_transform = target_transform
        self.stl_names = stl_names
        self.num_classes = len(stl_names)
        self.height = height
        self.width = width
        self.channels = channels
        self.windowlevel = windowlevel
        self.windowwidth = windowwidth
        self.status=status

        uids = []
        with open(uid_file) as f:
            uids = [x.strip() for x in f.readlines()]

        self.image_labels = []
        image_base_path = os.path.join(base_path, image_folder_name)
        for uid in uids:
            for filename in os.listdir(os.path.join(image_base_path, uid)):
                self.image_labels.append(os.path.join(uid, filename))
        # self.image_labels = self.image_labels[:50] ##debug
        print("uids:", len(uids), "image_label:", len(self.image_labels))

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        ## deal with image
        img_path = os.path.join(
            self.base_path, 
            self.image_folder_name, 
            self.image_labels[idx]
        )
        image = sitk.ReadImage(img_path)
        image = sitk.GetArrayFromImage(image)
        # image = ((image - self.windowlevel) / self.windowwidth + 0.5) * 255.0

        # ## clip to (0, 1)
        image = (image - self.windowlevel) / self.windowwidth + 0.5
        image = np.clip(image, 0, 1) * 255
        image = image.reshape((self.height, self.width, self.channels))

        ## deal with label
        mask = np.zeros((self.height, self.width, self.num_classes), dtype=float)
        for index, stl_name in enumerate(self.stl_names):
            label_path = os.path.join(
                self.base_path, 
                self.mask_folder_name, 
                stl_name, 
                self.image_labels[idx]
            )
            tmp = sitk.ReadImage(label_path)
            tmp = sitk.GetArrayFromImage(tmp)
            tmp = tmp.reshape((self.height, self.width))
            mask[:, :, index] = tmp
        # mask = mask.transpose(2, 0, 1)

        if self.status:
            ## Operate rotated must be applied to the HWC image, not the CHW image.
            def _rotate(image, angle=90):
                shape = image.shape
                (h, w) = shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(image, M, (w, h))
                rotated = rotated.reshape(shape)

                # HWC-->CHW
                # rotated = rotated.transpose(2, 0, 1)
                return rotated

            angle = random.randint(0, 8) * 45
            image = _rotate(image, angle)
            mask = _rotate(mask, angle)

        if self.transform:
            image = image.transpose(2, 0, 1).astype(np.uint8)
            mask = mask.transpose(2, 0, 1).astype(np.uint8)
            transform_image_mask = self.transform(image=image, mask=mask)
            image = transform_image_mask["image"]
            mask = transform_image_mask["mask"]
        else:
            image = image.transpose(2, 0, 1)
            mask = mask.transpose(2, 0, 1)
        
        ##ValueError: At least one stride in the given numpy array is negative, 
        # and tensors with negative strides are not currently supported. 
        # (You can probably work around this by making a copy of your array  with array.copy().)
        image = image.copy()
        mask = mask.copy()
        return image, mask


class SegDataset1(Dataset):
    def __init__(
        self,
        # uid_file, ## "/Data/data10/dst/50_10_增强全部件/train.txt"
        base_path, ## /Data/data10/dst/dicom/
        # model_name,
        stl_names=["lung"],
        height=512,
        width=512,
        channels=1,
        windowlevel=-600,
        windowwidth=2000,
        transform=None,
        target_transform=None,
        image_folder_name="dicom",
        mask_folder_name="mask",
        status=True, # True for training, False for test
    ):
        ## TO DO: 完成数据列表的统计，transform
        ## input：list_txt, 

        self.base_path = base_path
        # self.model_name = model_name
        self.image_folder_name = image_folder_name
        self.mask_folder_name = mask_folder_name
        self.transform = transform
        self.target_transform = target_transform
        self.stl_names = stl_names
        self.num_classes = len(stl_names)
        self.height = height
        self.width = width
        self.channels = channels
        self.windowlevel = windowlevel
        self.windowwidth = windowwidth

        self.min_bound = windowlevel - windowwidth // 2  
        self.max_bound = windowlevel + windowwidth // 2

        self.uids = os.listdir(base_path)
        self.image_labels = []
        self.status = status

        for uid in self.uids:
            self.image_path = os.path.join(base_path, uid, image_folder_name)
            image_names = os.listdir(self.image_path)

            self.mask_path = os.path.join(base_path, uid, mask_folder_name, "lung")
            mask_names = os.listdir(self.mask_path)
            for imagen, maskn in zip(image_names, mask_names):
                self.image_labels.append([os.path.join(self.image_path, imagen), 
                # os.path.join(self.mask_path, maskn)
                maskn  ## 只保存maskname
                ]) 
        # self.image_labels = self.image_labels[:50]  ###debug
        print("uids:", len(self.uids), "image_label:", len(self.image_labels))


    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        ## deal with image
        img_path = os.path.join(
            self.base_path,
            self.image_labels[idx][0]
        )
        image = sitk.ReadImage(img_path)
        image = sitk.GetArrayFromImage(image)
        ## method1
        # image = ((image - self.windowlevel) / self.windowwidth + 0.5) * 255.0

        ## method2
        image = ((image - self.windowlevel) / self.windowwidth + 0.5)
        image = np.clip(image, 0, 1) * 255.0

        # ## method3
        # image = (image - self.min_bound) / (self.max_bound - self.min_bound)
        # image = np.clip(image, 0, 1)

        image = image.reshape((self.height, self.width, self.channels))

        ## deal with label
        mask = np.zeros((self.height, self.width, self.num_classes), dtype=float)
        for index, stl_name in enumerate(self.stl_names):
            cur_uid = img_path.split('\\')[-3]
            label_path = os.path.join(
                self.base_path, 
                cur_uid,
                self.mask_folder_name, 
                stl_name, 
                self.image_labels[idx][1]
            )
            tmp = sitk.ReadImage(label_path)
            tmp = sitk.GetArrayFromImage(tmp)
            tmp = tmp.reshape((self.height, self.width))
            mask[:, :, index] = tmp 

        if self.status:
            ### Operate rotated Imust be applied to the HWC image, not the CHW image.
            def _rotate(image, angle=90):
                shape = image.shape
                (h, w) = shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(image, M, (w, h))
                rotated = rotated.reshape(shape)

                # HWC-->CHW
                # rotated = rotated.transpose(2, 0, 1)
                return rotated
            angle = random.randint(0, 8) * 45
            image = _rotate(image, angle)
            mask = _rotate(mask, angle)

        # if self.transform:
        #     image = self.transform(image) ## 旋转，reshape，astype
        #     # mask = self.transform(mask)
        # if self.target_transform:
        #     label = self.target_transform(label)

        
        if self.transform:
            image = image.transpose(2,0,1).astype(np.uint8)
            mask = mask.transpose(2,0,1).astype(np.uint8)
            transform_image_mask = self.transform(image=image, mask=mask)
            image = transform_image_mask["image"]
            mask = transform_image_mask["mask"]
        else:
            image = image.transpose(2,0,1)
            mask = mask.transpose(2,0,1)

        ##ValueError: At least one stride in the given numpy array is negative, 
        # and tensors with negative strides are not currently supported. 
        # (You can probably work around this by making a copy of your array  with array.copy().)
        image = image.copy()
        mask = mask.copy()
        return image, mask