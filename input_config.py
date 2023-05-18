
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

    # "windowlevel": 0,
    # "windowwidth": 2000,

    "windowlevel": [-600, 135, 50, -850],  ## lung, skin, heart
    "windowwidth": [2000, 385, 500, 310],

    "train_base_path": "D:\\project\\TrueHealth\\20230217_Alg1\\data\\examples\\src_seg\\train",
    "valid_base_path": "D:\\project\\TrueHealth\\20230217_Alg1\\data\\examples\\src_seg\\val",

    # "encoder_name": "efficientnet-b4",
    # "encoder_name": "mobileone_s4",
    # "encoder_name": "mobileone_s3",
    # "encoder_name": "resnext101_32x4d",
    # "encoder_name": "tu-regnety_040", #regnety_040
    # "encoder_name": "timm-regnety_040", #regnety_040
    "encoder_name": "stdc2", ## stdc2
    "decoder_name": "Unet", #"MANet", #
    "decoder_attention_type": None,
    "stragety": "clip-rotated-class3", #"normal-rotated", #
    "pretrained_model": None,
    # "pretrained_model": ".\output\pth\stdc2_Unet\clip-rotated-32x-customLR1\\0413_090007\stdc2_Unet_clip-rotated-32x-customLR1_epoch_22.pth",
    # "pretrained_model": ".\\output\pth\\efficientnet-b4_epoch_7.pth",
    # "pretrained_model": ".\\output\\pth\\resnext101_32x4d\\0321_092203\\resnext101_32x4d_epoch_30.pth",
    # "pretrained_model": ".\\output\\pth\\tu-regnety_040_MANet\\0321_172201\\tu-regnety_040_MANet_epoch_0.pth",
    # dataloader config

    
    "shuffle": True,  # 是否需要打乱数据
    "num_workers": 4,  # 多线程加载所需要的线程数目
    "pin_memory": True,  # 数据从CPU->pin_memory—>GPU加速

    # model config
    "classes": ["lung", "skin", "heart", "zhiqiguan"],  #["zhiqiguan"], #  ["lung"], #
    "output_stride": 32,
    "in_channels": 4,  ## CT  slice 
    "batch_size": 4, #16 (for stdc) 4 (for efficientnet)
    "middle_patch_size": 512,
    "patch_size": 299,
    "plot_every": 50,

    "loss_function": "bce", ## focal, dice, bce
    "mode": "binary", ## "multiclass", "multilabel"
    
    "optimizer_name": "adam", #"adamw",#"adam""sgd", #
    "weight_decay": 0, # 1e-5
    "eps": 1e-8,
    "scheduler_name": "customLR1", #"OneCycleLR", #

    # # RMS
    # "weight_decay": 0.9,
    "momentum": 0.9,
    # "eps": 1,  # DeepPATH  RMSPROP_EPSILON = 1.0

    # lr
    "gamma": 0.16,
    "step_size": 30, # lr will be decay every step_size epoch

    "max_epoch": 100,
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
