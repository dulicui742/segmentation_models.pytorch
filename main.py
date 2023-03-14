import os
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

    ## load model
    encoder_name= entrance["encoder_name"]
    model = smp.Unet(
        encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None, # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1, # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
    )

    print("load model...!")
    pretrain_model_path = entrance["pretrained_modle"]
    if pretrain_model_path is not None:
        state_dict = torch.load(pretrain_model_path)
        model.load_state_dict(state_dict)
        print("load model")
        initial_epoch = pretrain_model_path.split("_")[-1].split(".")[0]
    else:
       initial_epoch = 0 

    ## dataloader
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

    print("----------start to train------------")
    criterion = BCELoss()
    metrics = [IoU(), Fscore(), Precision(), Recall(), Accuracy()]
    device = torch.device(entrance["device"] if torch.cuda.is_available() else "cpu")
    train_obj = TrainEpoch(
        model,
        criterion,
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
        # logs = train_obj.run(dataloader)
        logs = train_obj.custom_run(dataloader, epoch)
        ## to save logs
        import pdb; pdb.set_trace()
        ## to save model
        save_path = os.path.join(
            entrance["save_base_path"], 
            "{}_epoch_{}.pth".format(encoder_name, epoch))
        torch.save(model.state_dict(), save_path)

        ## valid
        val_logs = valid_obj.custom_run(val_dataloader, epoch)



def main(entrance):
    train_dataset = SegDataset1(
        base_path=entrance["train_base_path"], transform=None
    )
    val_dataset = SegDataset1(
        base_path=entrance["valid_base_path"], transform=None
    )
    train(train_dataset, val_dataset)


if __name__ == "__main__":
    main(entrance)
    # x = torch.FloatTensor(np.ones([10,3,256,256])).cuda()
    # backbone= get_encoder("efficientnet-b4").cuda()

    # import pdb; pdb.set_trace()
    ## model
    # model = smp.Unet(
    #     encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    #     encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
    #     in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    #     classes=1,                      # model output channels (number of classes in your dataset)
    # )
    # # ff = model(x)
    # # import pdb; pdb.set_trace()

    # ## dataset
    # print("create dataset for training set and validation set!")
    # transform = transforms.Compose(
    #             [
    #                 # transforms.Resize((512, 512)),
    #                 # transforms.RandomCrop((224, 224)),
    #                 transforms.ToTensor(),  # 归一化到[0,1]
    #                 # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #归一化到[-1,1]
    #                 ## channel = (channel-mean) / std(因为transforms.ToTensor()已经把数据处理成[0,1],那么(x-0.5)/0.5就是[-1.0, 1.0])
    #                 ##这样一来，我们的数据中的每个值就变成了[-1,1]的数了。
    #             ]
    #         )
    # dataset = SegDataset1(
    #     # uid_file, ## "/Data/data10/dst/50_10_增强全部件/train.txt"
    #     base_path="D:\\project\\TrueHealth\\20230217_Alg1\\data\\examples\\src_seg\\train", ## /Data/data10/dst/dicom/
    #     # model_name,
    #     # stl_names,
    #     # transform=transform
    # )
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     4, ##batch_szie
    #     num_workers=8,
    #     shuffle=True,
    #     pin_memory=True,
    #     drop_last=False,
    # )


    # val_dataset = SegDataset1(
    #     # uid_file, ## "/Data/data10/dst/50_10_增强全部件/train.txt"
    #     base_path="D:\\project\\TrueHealth\\20230217_Alg1\\data\\examples\\src_seg\\val", ## /Data/data10/dst/dicom/
    #     # model_name,
    #     # stl_names,
    #     # transform=transform
    # )
    # val_dataloader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     4, ##batch_szie
    #     num_workers=8,
    #     shuffle=False,
    #     pin_memory=True,
    #     drop_last=False,
    # )

    # # for ii, (dicom, label) in enumerate(dataloader):
    # #     # pdb.set_trace()
    # #     print("ii:", ii, dicom.shape, label.shape)
    # #     if ii > 5:
    # #         break

    # ## optimizer
    # initial_epoch = 0
    # epochs = 3
    # save_base_path = "D:\\project\\TrueHealth\\git\\segmentation_models.pytorch\\output\\pth"
    
    # weight_decay = 1e-5
    # eps = 1e-8
    # momentum = 0.9
    # lr = 1e-4
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # opts = {
    #     "sgd": get_sgd_optimizer(
    #         # model, lr, weight_decay=1e-5
    #         model, lr, momentum=momentum, weight_decay=weight_decay
    #     ),
    #     "rms": get_rms_optimizer(
    #         # model, lr, momentum, weight_decay=1e-5
    #         model, lr, momentum=momentum, weight_decay=weight_decay, eps=eps
    #     ),
    #     "adam": get_adam_optimizer(
    #         # model, lr, alpha=0.9, momentum=0.9, weight_decay=1e-5, eps=1e-08
    #         model, lr, weight_decay=weight_decay,
    #     ),
    # }

    # ### loss
    # import pdb; pdb.set_trace()
    # # criterion = SoftBCEWithLogitsLoss(smooth_factor=0.1, ignore_index=-100)
    # criterion = BCELoss()
    # metrics = [IoU(), Fscore(), Precision(), Recall(), Accuracy()]
    # train_obj = TrainEpoch(
    #     model,
    #     criterion,
    #     metrics,
    #     opts["adam"], ## optimizer
    #     device=device,
    # )

    # valid_obj = ValidEpoch(
    #     model,
    #     criterion,
    #     metrics,
    #     device=device,
    # )

    # ## start to train
    # ## load model params
    # # if initial_epoch != 0:  # 如果有预训练模型则加载预训练模型
    # #     state_dict = torch.load(pretrain_model_path)
    # #     model.load_state_dict(state_dict)
    # #     print("load model")

    # for epoch in range(initial_epoch, epochs):
    #     # import pdb; pdb.set_trace()
    #     # logs = train_obj.run(dataloader)
    #     logs = train_obj.custom_run(dataloader, epoch)
    #     ## to save logs
    #     # import pdb; pdb.set_trace()
    #     ## to save model
    #     save_path = os.path.join(save_base_path, "eunet_epoch{}.pth".format(epoch))
    #     torch.save(model.state_dict(), save_path)

    #     ## valid
    #     val_logs = valid_obj.custom_run(val_dataloader, epoch)



    print("end")
