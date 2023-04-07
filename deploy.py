import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import time
import torch
import numpy as np
import vtk
import itk

from vtk.util import numpy_support
from vtk import vtkMatrix4x4

import albumentations as A

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.datasets import(
    OxfordPetDataset, 
    SimpleOxfordPetDataset,
    SegDataset,
    SegDataset1
)
from segmentation_models_pytorch.utils.train import(
    Epoch, TrainEpoch, ValidEpoch
)


def visualize(images):
    rows = len(images)
    n = 3 ## image, gt, pred
    plt.figure(figsize=(16, 5))

    for j in range(rows):
        for i, (name, image) in enumerate(images[j].items()):
            plt.subplot(rows, n, j * n + i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image, cmap=plt.cm.gray)
    # plt.show()
    plt.pause(1)


def load_model(encoder_name, model_path, device, in_channels=1, classes=1):
    ## 实例化
    model = smp.Unet(
    # model = smp.MAnet(
        encoder_name=encoder_name,  
        encoder_weights=None, 
        in_channels=in_channels, 
        classes=classes,
    ).to(device)

    ## load model parameters
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict) # strict=False
    return model


def deploy_model(dataset, **entrance):
    encoder_name = entrance["encoder_name"]
    device = torch.device(entrance["device"] if torch.cuda.is_available() else "cpu")
    model_path = entrance["best_model"]
    save_base_path = entrance["save_base_path"]

    model = load_model(encoder_name, model_path, device, in_channels=1, classes=1)
    model.encoder.set_swish(memory_efficient=False)
    model.eval()

    np.random.seed(1)
    idx = np.random.randint(0, len(dataset))
    print("idx", idx)

    # import pdb; pdb.set_trace()
    image, gt_mask = dataset[idx]
    x_tensor = torch.from_numpy(image).to(device).unsqueeze(0).float()

    with torch.no_grad():
        trace_model = torch.jit.trace(model, x_tensor)
        # trace_model = torch.jit.script(model, x_tensor)

        pt_save_base_path = os.path.join(save_base_path, "pt")
        if not os.path.exists(pt_save_base_path):
            os.makedirs(pt_save_base_path)
        model_name = os.path.splitext(model_path.split('\\')[-1])[0]
        pt_name = os.path.join(pt_save_base_path, "{}_trace.pt". format(model_name))
        print(pt_name)
        # import pdb; pdb.set_trace()
        trace_model.save(pt_name)

    pred1 = model.predict(x_tensor)
    pred2 = trace_model(x_tensor)

    # import pdb; pdb.set_trace()
    print("Happy End!")


def test(test_dataset, **entrance):
    save_base_path = entrance["save_base_path"]
    model_path = entrance["best_model"]
    device = torch.device(entrance["device"] if torch.cuda.is_available() else "cpu")

    ## load pt model
    pt_save_base_path = os.path.join(save_base_path, "pt")
    model_name = os.path.splitext(model_path.split('\\')[-1])[0]
    pt_name = os.path.join(pt_save_base_path, "{}_trace.pt". format(model_name))
    best_model = torch.jit.load(pt_name)

    index = random.sample(range(30, len(test_dataset)), 20)
    for i in index:
        n = i
        image, gt_mask = test_dataset[n]
        # gt_mask = gt_mask.squeeze()
        
        x_tensor = torch.from_numpy(image).to(device).unsqueeze(0).float()
        pr_mask = best_model(x_tensor)  ## call model.forward()

        pr_mask = torch.sigmoid(pr_mask)
        pr_mask = (pr_mask.squeeze(0).cpu().detach().numpy())

        image = image.squeeze().astype(np.uint8) 
        params = []
        for i in range(len(classes)):
            params.append({
                "image": image, 
                "gt_{}" .format(classes[i]): gt_mask[i,:,:], 
                "pred_{}" .format(classes[i]): pr_mask[i,:,:]})
        visualize(params)


if __name__ == "__main__":
    entrance = {
        "encoder_name": "efficientnet-b4",
        # "best_model": "/home/th/Data/dulicui/share/efficientnet-b4_MANet_epoch_41.pth", #21， 41， 56
        "best_model": "D:\\share\\deploy\\Bronchial\\efficientnet-b4_epoch_23.pth", #21 0321-151721
        
        "decoder_name": "Unet", #"MANet", #
        "device": "cuda:0",
        "test_uid_file": "D:\\project\\TrueHealth\\20230217_Alg1\\data\\examples\\dst\\imageset\\test.txt",
        "test_base_path": "D:\\project\\TrueHealth\\20230217_Alg1\\data\\examples\\dst",
        "windowlevel": -850,
        "windowwidth": 310,
        "middle_patch_size": 512,
        "classes": ["Bronchial"],
        "in_channels": 1,
        "num_workers": 4,  # 多线程加载所需要的线程数目
        "pin_memory": True,  # 数据从CPU->pin_memory—>GPU加速
        "batch_size": 4,
        "save_base_path": "D:\\share\\deploy\\Bronchial",
    }

    test_dataset = smp.datasets.SegDataset(
        uid_file=entrance["test_uid_file"],
        base_path=entrance["test_base_path"],
        stl_names=entrance["classes"],
        height=entrance["middle_patch_size"],
        width=entrance["middle_patch_size"],
        # channels=entrance["in_channels"],
        windowlevel=entrance["windowlevel"],
        windowwidth=entrance["windowwidth"],
        transform=None,
        # transform=transform,
        status=False
    )

    # deploy_model(test_dataset, **entrance)
    test(test_dataset, **entrance)



