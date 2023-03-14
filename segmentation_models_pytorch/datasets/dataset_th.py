import os
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
        model_name,
        stl_names,
        height=512,
        width=512,
        windowlevel=-600,
        windowwidth=2000,
        transform=None,
        target_transform=None,
        image_folder_name="dicom",
        mask_folder_name="mask",
    ):
        ## TO DO: 完成数据列表的统计，transform
        ## input：list_txt, 

        self.base_path = base_path
        self.model_name = model_name
        self.image_folder_name = image_folder_name
        self.mask_folder_name = mask_folder_name
        self.transform = transform
        self.target_transform = target_transform
        self.stl_names = stl_names
        self.num_classes = len(stl_names)
        self.height = height
        self.width = width
        self.windowlevel = windowlevel
        self.windowwidth = windowwidth

        uids = []
        with open(uid_file) as f:
            uids.append(x.strip() for x in f)

        self.image_labels = []
        for uid in uids:
            for filename in os.listdir(uid):
                self.image_labels.append(os.path.join(uid, filename))


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
        image = ((image - self._windowlevel) / self._windowwidth + 0.5) * 255.0


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
            tmp = img.reshape((self.height, self.width))
            mask[:, :, index] = tmp
        
        # if self.target_transform:
        #     label = self.target_transform(label)
        if self.transform:
            image = self.transform(image) ## 旋转，reshape，astype
            mask = self.transform(mask)
        return image, mask


class SegDataset1(Dataset):
    def __init__(
        self,
        # uid_file, ## "/Data/data10/dst/50_10_增强全部件/train.txt"
        base_path, ## /Data/data10/dst/dicom/
        # model_name,
        # stl_names,
        height=512,
        width=512,
        windowlevel=-600,
        windowwidth=2000,
        transform=None,
        target_transform=None,
        image_folder_name="dicom",
        mask_folder_name="mask",
    ):
        ## TO DO: 完成数据列表的统计，transform
        ## input：list_txt, 

        self.base_path = base_path
        # self.model_name = model_name
        self.image_folder_name = image_folder_name
        self.mask_folder_name = mask_folder_name
        self.transform = transform
        self.target_transform = target_transform
        # self.stl_names = stl_names
        self.height = height
        self.width = width
        self.windowlevel = windowlevel
        self.windowwidth = windowwidth

        self.uids = os.listdir(base_path)
        self.image_labels = []

        # import pdb; pdb.set_trace()
        for uid in self.uids:
            self.image_path = os.path.join(base_path, uid, image_folder_name)
            image_names = os.listdir(self.image_path)

            self.mask_path = os.path.join(base_path, uid, mask_folder_name, "fei")
            mask_names = os.listdir(self.mask_path)
            for imagen, maskn in zip(image_names, mask_names):
                self.image_labels.append([os.path.join(self.image_path, imagen), 
                os.path.join(self.mask_path, maskn)])  
        print("uids:", len(self.uids), "image_label:", len(self.image_labels))


    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        ## deal with image
        img_path = self.image_labels[idx][0]
        image = sitk.ReadImage(img_path)
        image = sitk.GetArrayFromImage(image)
        # print(image.shape)
        image = ((image - self.windowlevel) / self.windowwidth + 0.5) * 255.0
        # image.astype(np.float32)

        ## deal with label
        mask = np.zeros((self.height, self.width, 1), dtype=float)
        label_path = self.image_labels[idx][1]
        # print("img:", img_path, "mask:", label_path)
        tmp = sitk.ReadImage(label_path)
        tmp = sitk.GetArrayFromImage(tmp)
        # print(tmp.shape)
        tmp = tmp.reshape((self.height, self.width))
        mask[:, :, 0] = tmp
        mask = mask.transpose(2,0,1)
        
        
        if self.transform:
            image = self.transform(image) ## 旋转，reshape，astype
            # mask = self.transform(mask)
        if self.target_transform:
            label = self.target_transform(label)

        # import pdb; pdb.set_trace()
        # image = image.float()
        # mask = mask.float()
        # print(image.dtype, mask.shape)
        return image, mask