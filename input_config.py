
# encoding:utf-8
import time
import os
from functools import partial
import torchvision.transforms as transforms


# tfmt = '%m%d_%H%M%D'
tfmt = "%m%d"
entrance = {
    # "windowlevel": -600,
    # "windowwidth": 2000,
    # "windowlevel": -850,
    # "windowwidth": 310,

    "windowlevel": 0,
    "windowwidth": 2000,

    "train_base_path": "D:\\project\\TrueHealth\\20230217_Alg1\\data\\examples\\src_seg\\train",
    "valid_base_path": "D:\\project\\TrueHealth\\20230217_Alg1\\data\\examples\\src_seg\\val",

    # "encoder_name": "efficientnet-b4",
    # "encoder_name": "mobileone_s4",
    # "encoder_name": "mobileone_s3",
    # "encoder_name": "resnext101_32x4d",
    # "encoder_name": "tu-regnety_040", #regnety_040
    # "encoder_name": "timm-regnety_040", #regnety_040
    "encoder_name": "stdc1", ## stdc2
    "decoder_name": "Unet", #"MANet", #
    "stragety": "clip-rotated", #"normal-rotated", #
    "pretrained_model": None,
    # "pretrained_model": ".\\output\pth\\efficientnet-b4_epoch_7.pth",
    # "pretrained_model": ".\\output\\pth\\resnext101_32x4d\\0321_092203\\resnext101_32x4d_epoch_30.pth",
    # "pretrained_model": ".\\output\\pth\\tu-regnety_040_MANet\\0321_172201\\tu-regnety_040_MANet_epoch_0.pth",
    # dataloader config
    "shuffle": True,  # 是否需要打乱数据
    "num_workers": 4,  # 多线程加载所需要的线程数目
    "pin_memory": True,  # 数据从CPU->pin_memory—>GPU加速

    # model config
    "classes": ["lung"], #["lung", "skin", "heart"],   #["zhiqiguan"], #
    "output_stride": 32,
    "in_channels": 1,  ## CT  slice 
    "batch_size": 16, #16 (for stdc) 4 (for efficientnet)
    "middle_patch_size": 512,
    "patch_size": 299,
    "plot_every": 50,

    "optimizer_name": "adam", #"adamw",#
    "weight_decay": 1e-5, # 
    "eps": 1e-8,
    "scheduler_name": "OneCycleLR", #"customLR1", #

    # # RMS
    # "weight_decay": 0.9,
    "momentum": 0.9,
    # "eps": 1,  # DeepPATH  RMSPROP_EPSILON = 1.0

    # lr
    "gamma": 0.16,
    "step_size": 30, # lr will be decay every step_size epoch

    "max_epoch": 60,
    "lr": 1e-4,  # 学习率
    "min_lr": 1e-10,  # 当学习率低于这个值，就退出训练1e-10
    "lr_decay": 0.5,  # 当一个epoch的损失开始上升lr = lr*lr_decay

    # output path
    "temp_time": time.strftime(tfmt),
    "save_base_path": '.\\output',
    "pth_folder": "pth",
    "log_folder": "logs",

    # # log_name
    # "log_name": "log.txt"

    "device": "cuda:0"
}
