import os
import time
import torch 
import numpy as np
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
    BCELoss
)
from segmentation_models_pytorch.utils.metrics import(
    IoU, Fscore, Precision, Recall, Accuracy
)
from segmentation_models_pytorch.utils.optimizers import(
    get_adam_optimizer,
    # get_yellow,
    get_sgd_optimizer,
    get_rms_optimizer
)
from input_config import entrance


def parse(kwargs):
    ## 处理配置参数
    for k, v in kwargs.items():
        if not hasattr(entrance, k):
            print("Warning: opt has not attribut %s" % k)
        setattr(entrance, k, v)
    for k, v in entrance.__class__.__dict__.items():
        if not k.startswith("__"):
            print(k, getattr(entrance, k))


def train(train_dataset, val_dataset, **kwargs):
    parse(kwargs)

    ## ===============define model================
    tfmt = "%m%d"
    encoder_name= entrance["encoder_name"]
    num_classes = len(entrance["label_map"])
    model = smp.Unet(
        encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None, # use `imagenet` pre-trained weights for encoder initialization
        # in_channels=entrance["in_channels"], # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        # classes=num_classes,  # model output channels (number of classes in your dataset)
        in_channels=1, # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,
    )

    ## ===============load pretrained model===============
    print("load model...!")
    pretrain_model_path = entrance["pretrained_modle"]
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

    ##=====================optimizer=================
    pre_loss = 100
    lr = entrance["lr"]
    weight_decay = entrance["weight_decay"]
    momentum = entrance["momentum"]
    eps = entrance["eps"]
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
    }
    optimizer = optimizers[entrance["optimizer_name"]]

    ## ==================start to train===================
    print("----------start to train------------")
    criterion = BCELoss()
    metrics = [IoU(), Fscore(), Precision(), Recall(), Accuracy()]
    device = torch.device(entrance["device"] if torch.cuda.is_available() else "cpu")
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
    for epoch in range(initial_epoch, entrance["max_epoch"]):
        # import pdb; pdb.set_trace()
        print('\nEpoch: {}'.format(epoch))
        log_save_base_path = os.path.join(
            entrance["save_base_path"],
            entrance["log_folder"],
            encoder_name
        )
        if not os.path.exists(log_save_base_path):
            os.makedirs(log_save_base_path)

        log_filename = os.path.join(
            log_save_base_path,
            "{}_logs_{}.json".format(encoder_name, time.strftime(tfmt))
        )
        # logs = train_epoch.run(dataloader)
        train_logs = train_epoch.custom_run(dataloader, epoch, log_filename)

        ## to save model
        pth_save_base_path = os.path.join(
            entrance["save_base_path"], 
            entrance["pth_folder"],
            encoder_name
        )
        if not os.path.exists(pth_save_base_path):
            os.makedirs(pth_save_base_path)
        pth_filename = os.path.join(
            pth_save_base_path,
            "{}_epoch_{}.pth".format(encoder_name, epoch)
        )
        torch.save(model.state_dict(), pth_filename)

        ## valid
        valid_logs = valid_obj.custom_run(val_dataloader, epoch, log_filename)

        # keras.callbacks.ReduceLROnPlateau
        # torch.optim.lr_scheduler.ReduceLROnPlateau
        # if loss_meter.value()[0] > pre_loss * 1.0:
        if train_logs["bce_loss"] > pre_loss * 1.0:
            old_lr = lr
            lr = lr * entrance["lr_decay"]
            print("lr decay called: from {} to {}" .format(old_lr, lr))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        pre_loss = train_logs["bce_loss"] #loss_meter.value()[0]
        if lr < entrance["min_lr"]:
            break


def main(entrance):
    train_dataset = SegDataset1(
        base_path=entrance["train_base_path"],
        height=entrance["middle_patch_size"],
        width=entrance["middle_patch_size"],
        windowlevel=entrance["windowlevel"],
        windowwidth=entrance["windowwidth"],
        transform=None
    )
    val_dataset = SegDataset1(
        base_path=entrance["valid_base_path"],
        height=entrance["middle_patch_size"],
        width=entrance["middle_patch_size"],
        windowlevel=entrance["windowlevel"],
        windowwidth=entrance["windowwidth"],
        transform=None
    )
    train(train_dataset, val_dataset)


if __name__ == "__main__":
    main(entrance)
    print("end")