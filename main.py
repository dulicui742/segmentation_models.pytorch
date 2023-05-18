import os
import time
import torch 
import numpy as np
import albumentations as A
import torchvision.transforms as transforms

import segmentation_models_pytorch as smp 
from segmentation_models_pytorch.encoders import (
    efficient_net_encoders, 
    get_encoder, 
    get_encoder_names
)
from segmentation_models_pytorch.datasets import(
    OxfordPetDataset, 
    SimpleOxfordPetDataset,
    SegDataset,
    SegDataset1
)
from segmentation_models_pytorch.utils.train import(
    Epoch, TrainEpoch, ValidEpoch
)
# from segmentation_models_pytorch.losses import (
#     DiceLoss,
#     JaccardLoss,
#     SoftBCEWithLogitsLoss,
#     SoftCrossEntropyLoss,
#     TverskyLoss,
#     MCCLoss,
# )
from segmentation_models_pytorch.utils.losses import(
    JaccardLoss,
    DiceLoss,
    DiceLoss1,
    BCELoss,
    BCEWithLogitsLoss,
    FocalLoss,
)
from segmentation_models_pytorch.utils.metrics import(
    IoU, Fscore, Precision, Recall, Accuracy
)
from segmentation_models_pytorch.utils.optimizers import(
    get_adam_optimizer,
    # get_yellow,
    get_sgd_optimizer,
    get_rms_optimizer,
    get_adamW_optimizer,
)
from segmentation_models_pytorch.utils.customLR import Custom1 as CustomLR1
from segmentation_models_pytorch.utils.base import SumOfLosses
from input_config import entrance as cfg1
from input_config2 import Config as cfg2


def parse(kwargs):
    ## 处理配置参数
    for k, v in kwargs.items():
        if not hasattr(entrance, k):
            print("Warning: opt has not attribut %s" % k)
        setattr(entrance, k, v)
    for k, v in entrance.__class__.__dict__.items():
        if not k.startswith("__"):
            print(k, getattr(entrance, k))


def train(train_dataset, val_dataset, **entrance):
    ## ===============define model================
    tfmt = "%m%d_%H%M%S" #"%m%d"
    encoder_name = entrance["encoder_name"]
    decoder_name = entrance["decoder_name"]
    stragety = entrance["stragety"]
    output_stride = entrance["output_stride"]
    num_classes = len(entrance["classes"])
    decoder_attention_type = entrance["decoder_attention_type"]
    in_channels = entrance["in_channels"]
    
    if decoder_name == "Unet" or decoder_name == "AttentionUnet":
        kwargs = {
        "output_stride": output_stride, 
        "decoder_attention_type": decoder_attention_type, 
    }
    else:
        kwargs = {}

    model = smp.create_model(
        arch=decoder_name,
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        classes=num_classes,
        **kwargs,
    )

    ## ===============load pretrained model===============
    print("load model...!")
    pretrain_model_path = entrance["pretrained_model"]
    if pretrain_model_path is not None:
        state_dict = torch.load(pretrain_model_path)
        model.load_state_dict(state_dict)
        print("load model")
        initial_epoch = int(pretrain_model_path.split("_")[-1].split(".")[0])
    else:
       initial_epoch = 0 

    ## ================dataloader==================
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        entrance["batch_size"], ##batch_szie
        num_workers=entrance["num_workers"],
        shuffle=entrance["shuffle"],
        pin_memory=entrance["pin_memory"],
        drop_last=False,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        entrance["batch_size"], ##batch_szie
        num_workers=entrance["num_workers"],
        shuffle=False,
        pin_memory=entrance["pin_memory"],
        drop_last=False,
    )

    ##=====================optimizer/loss=================
    pre_loss = 100
    lr = entrance["lr"]
    lr_decay = entrance["lr_decay"]
    weight_decay = entrance["weight_decay"]
    momentum = entrance["momentum"]
    eps = entrance["eps"]
    max_epoch = entrance["max_epoch"]
    scheduler_name = entrance["scheduler_name"]
    mode = entrance["mode"]
    optimizer_name = entrance["optimizer_name"]

    optimizers = {
        "sgd": get_sgd_optimizer(
            model, lr, momentum=momentum, weight_decay=weight_decay
        ),
        "rms": get_rms_optimizer(
            model, lr, momentum=momentum, weight_decay=weight_decay, eps=eps
        ),
        "adam": get_adam_optimizer(
            model, lr, weight_decay=weight_decay,
        ),
        "adamw": get_adamW_optimizer(
            model, lr, weight_decay=weight_decay,
        )
    }
    optimizer = optimizers[optimizer_name]
    if scheduler_name == "customLR1":
        scheduler = CustomLR1(
            lr=lr,
            pre_loss=pre_loss,
            optimizer=optimizer,
            lr_decay=entrance["lr_decay"],
            min_lr=entrance["min_lr"],
        )
    elif scheduler_name == "Cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epoch
        )
    elif scheduler_name == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, steps_per_epoch=len(dataloader), epochs=max_epoch
        )
    elif scheduler_name == "StepLR":
        torch.optim.lr_scheduler.StepLR(
            optimizer, step_size, gamma=gamma, last_epoch=-1
        )
    elif scheduler_name == "exponentialLR":
        torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    else:
        print("Please provide scheduler for LR!!!")
    # print(scheduler)


    criterions = {
        "bce": BCELoss(), #BCEWithLogitsLoss(), 
        "dice": DiceLoss1(mode=mode,),
        "focal": FocalLoss(mode=mode, alpha=0.25,),
        "wbce": BCEWithLogitsLoss(pos_weight=torch.tensor([10])),
        "dice-bce": SumOfLosses(DiceLoss1(mode=mode,), BCEWithLogitsLoss()),
        "dice-focal": SumOfLosses(DiceLoss1(mode=mode,), FocalLoss(mode=mode, alpha=0.25,))
    }

    ## ==================start to train===================
    print("----------start to train------------")
    loss_function = entrance["loss_function"]
    criterion = criterions[loss_function]
    metrics = [IoU(), Fscore(), Precision(), Recall(), Accuracy()]
    device = torch.device(entrance["device"] if torch.cuda.is_available() else "cpu")

    print(f"lr: {lr}, lr_decay: {lr_decay}, momentum: {momentum}, weight_decay: {weight_decay},\
        loss: {criterion}, optimizer: {optimizer}, scheduler: {scheduler}"
    ) 

    train_epoch = TrainEpoch(
        model,
        criterion, #loss
        metrics,
        optimizer, ## optimizer
        device=device,
    )

    valid_obj = ValidEpoch(
        model,
        criterion,
        metrics,
        device=device,
    )

    env = entrance["env"]   
    if env:
        timestamp = entrance["timestamp"]
        log_save_base_path = entrance["log_save_base_path"]
        pth_save_base_path = os.path.join(entrance["pth_save_base_path"], timestamp)
    else:
        timestamp = time.strftime(tfmt)
        log_save_base_path = os.path.join(
            entrance["save_base_path"],
            "{}_{}". format(encoder_name, decoder_name),
            "{}-{}-{}x" .format(stragety, scheduler_name, output_stride),
            entrance["log_folder"],
        )
        pth_save_base_path = os.path.join(
        entrance["save_base_path"], 
        "{}_{}". format(encoder_name, decoder_name),
        "{}-{}-{}x" .format(stragety, scheduler_name, output_stride),
        entrance["pth_folder"],
        timestamp
    )

    if not os.path.exists(log_save_base_path):
        os.makedirs(log_save_base_path)
    if not os.path.exists(pth_save_base_path):
        os.makedirs(pth_save_base_path)

    log_filename = os.path.join(
        log_save_base_path,
        # "{}_{}_{}-{}x-{}_logs_{}.json".format(
        #     encoder_name, decoder_name, stragety, output_stride, scheduler_name, timestamp
        # )
        f"logs_{timestamp}.json",
    )

    for epoch in range(initial_epoch, max_epoch):
        print('\nEpoch: {}'.format(epoch))
        print("current lr: {}". format(scheduler.get_lr()))

        # logs = train_epoch.run(dataloader)
        train_logs = train_epoch.custom_run(dataloader, epoch, log_filename)

        ## to save model
        pth_filename = os.path.join(
            pth_save_base_path,
            # "{}_{}_{}-{}x-{}_epoch_{}.pth".format(
            #     encoder_name, decoder_name, stragety, output_stride, scheduler_name, epoch
            # )
            f"epoch_{timestamp}_{epoch}",
        )
        torch.save(model.state_dict(), pth_filename)

        ## valid
        valid_logs = valid_obj.custom_run(val_dataloader, epoch, log_filename)

        if isinstance(scheduler, CustomLR1):
            scheduler.step(train_logs[criterion.__name__])
        else:
            scheduler.step()

        # custom_scheduler
        print("current lr: {}". format(scheduler.get_lr()))
        if hasattr(scheduler, "should_break") and scheduler.should_break():
            print(f"Break because {scheduler} said so.")
            break


def main(entrance):
    width, heigth = entrance["middle_patch_size"], entrance["middle_patch_size"]
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([
            # noise
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.RandomContrast(),
            A.RandomGamma(),
            A.RandomBrightness(),
        ], p=0.2),
        A.OneOf([
            # blur
            # A.MotionBlur(p=0.2),
            A.GaussianBlur(p=0.1),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.OneOf([
            A.ElasticTransform(alpha=width * 2, sigma=width * 0.08, alpha_affine=width * 0.08),
            A.GridDistortion(),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5),
        ], p=0.2),
        A.ShiftScaleRotate(rotate_limit=180, p=0.5),
    ])
    
    train_dataset = SegDataset1(
        base_path=entrance["train_base_path"],
        stl_names=entrance["classes"],
        height=entrance["middle_patch_size"],
        width=entrance["middle_patch_size"],
        windowlevel=entrance["windowlevel"],
        windowwidth=entrance["windowwidth"],
        transform=None,
        # transform=transform,
        # status=False,
        is_multilabels=True
    )
    val_dataset = SegDataset1(
        base_path=entrance["valid_base_path"],
        stl_names=entrance["classes"],
        height=entrance["middle_patch_size"],
        width=entrance["middle_patch_size"],
        windowlevel=entrance["windowlevel"],
        windowwidth=entrance["windowwidth"],
        transform=None,
        # transform=transform,
        is_multilabels=True
    )
    train(train_dataset, val_dataset, **entrance)


if __name__ == "__main__":
    # if os.environ["USE_JSON_CONFIG_FILE"]:
    #     opt = cfg2.init_from_env()
    #     entrance = opt.to_dict()
    #     entrance["env"] = True
    # else:
    #     entrance = cfg1
    #     entrance["env"] = False

    entrance = cfg1
    entrance["env"] = False
    main(entrance)
    print("end")