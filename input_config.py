
# encoding:utf-8
import time
import os
from functools import partial
import torchvision.transforms as transforms


# tfmt = '%m%d_%H%M%D'
tfmt = "%m%d"
entrance = {
    "label_map": str(
        {
            # "backgrund": 0,
            "lung": 1,
        }
    ),

    "windowlevel": -600,
    "windowwidth": 2000,

    "train_base_path": "D:\\project\\TrueHealth\\20230217_Alg1\\data\\examples\\src_seg\\train",
    "valid_base_path": "D:\\project\\TrueHealth\\20230217_Alg1\\data\\examples\\src_seg\\val",

    "encoder_name": "efficientnet-b4",
    # "encoder_name": "mobileone_s4",
    "pretrained_modle": None,
    # "pretrained_modle": ".\\output\pth\\efficientnet-b4_epoch_7.pth",

    # dataloader config
    "shuffle": True,  # 是否需要打乱数据
    "num_workers": 8,  # 多线程加载所需要的线程数目
    "pin_memory": True,  # 数据从CPU->pin_memory—>GPU加速

    # model config
    "classes": ["lung"],
    "in_channels": 1,  ## CT  slice 
    "batch_size": 4,
    "middle_patch_size": 512,
    "patch_size": 299,
    "plot_every": 50,

    "optimizer_name": "adam",
    "weight_decay": 0, # 
    "eps": 1e-8,

    # # RMS
    # "weight_decay": 0.9,
    "momentum": 0.9,
    # "eps": 1,  # DeepPATH  RMSPROP_EPSILON = 1.0

    # lr
    "gamma": 0.16,
    "step_size": 30, # lr will be decay every step_size epoch

    "max_epoch": 300,
    "lr": 1e-4,  # 学习率
    "min_lr": 1e-10,  # 当学习率低于这个值，就退出训练
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
