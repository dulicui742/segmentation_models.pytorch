import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import torch
import numpy as np
import vtk

from vtk.util import numpy_support

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


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
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
    model.load_state_dict(state_dict, strict=False)
    return model


def test(test_dataset, **entrance):
    start_time = time.time()
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        entrance["batch_size"], ##batch_szie
        num_workers=entrance["num_workers"],
        shuffle=False,
        pin_memory=entrance["pin_memory"],
        drop_last=False,
    )

    encoder_name = entrance["encoder_name"]
    device = torch.device(entrance["device"] if torch.cuda.is_available() else "cpu")
    model_path = entrance["best_model"]
    model = load_model(encoder_name, model_path, device, in_channels=1, classes=1)

    ### re-parameterizable
    if  "mobileone" in encoder_name:
        print("~~~~~~~~~~~Re-parameter starts!~~~~~~~~~~")
        from segmentation_models_pytorch.encoders.mobileone import reparameterize_model
        best_model = reparameterize_model(model) ## 
    else:
        best_model = model

    model_time = time.time()
 
    # evaluate model on test set
    # for i in range(len(test_dataset)): ##从28个slice开始有肺
    # for i in range(100, 120): #28, 40
    for i in range(28, 291, 15):
    # for i in [68, 138]:
        n = i
        # n = np.random.choice(len(test_dataset))
        image, gt_mask = test_dataset[n]
        
        gt_mask = gt_mask.squeeze()
        
        x_tensor = torch.from_numpy(image).to(device).unsqueeze(0).float()
        pr_mask = best_model.predict(x_tensor)

        pr_mask = torch.sigmoid(pr_mask)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        image = image.squeeze().astype(np.uint8) 
        params = {"image": image, "ground_truth_mask": gt_mask, "predicted_mask": pr_mask}
        visualize(**params)

    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("model time:", model_time - start_time)
    print("inference time:", time.time() - start_time)


def generate_stl(**entrance):
    start_time = time.time()
    encoder_name = entrance["encoder_name"]
    decoder_name = entrance["decoder_name"]
    image_path = entrance["image_path"]
    device = torch.device(entrance["device"] if torch.cuda.is_available() else "cpu")
    windowlevel = entrance["windowlevel"]
    windowwidth = entrance["windowwidth"]
    best_model_path = entrance["best_model"]
    labels = entrance["classes"]

    best_model = load_model(encoder_name, best_model_path, device)
    load_model_time = time.time()

    dicomreader = vtk.vtkDICOMImageReader()
    dicomreader.SetDirectoryName(image_path)
    dicomreader.Update()
    output = dicomreader.GetOutput()
    dimensions = output.GetDimensions()
    print("dimension:", dimensions)

    # import pdb; pdb.set_trace()
    dicomArray = numpy_support.vtk_to_numpy(output.GetPointData().GetScalars())
    dicomArray = dicomArray.reshape(dimensions[::-1]).astype(np.float32)
    copyArray = dicomArray * 0

    # import pdb; pdb.set_trace()
    for i in range(dimensions[2]):
        img = dicomArray[i, ::-1, :]  # slice
        img = ((img - windowlevel) / windowwidth + 0.5) * 255
        # cv2.imshow("img", img/255)
        # params = {"image": img}
        # visualize(**params)
        img = img.reshape((1, dimensions[1], dimensions[0]))
        x_tensor = torch.from_numpy(img).to(device).unsqueeze(0).float()
        preds = best_model.predict(x_tensor)  # 预测图
        # preds = preds.permute(0, 2, 3, 1).cpu()
        # preds = torch.sigmoid(preds)
        (n, c, h, w) = preds.shape
        # preds = (preds.squeeze().cpu().numpy().round()) ## shape: 512 * 512
        preds = (preds.squeeze().cpu().numpy())

        preds[preds > 0.1] = 1
        # params = {"pred0": preds}
        # visualize(**params)
        tmp = copyArray[i, ::-1, :] + preds * 1 #pix
        copyArray[i, ::-1, :] += np.clip(tmp, 0, 1) #pix
        # copyArray[i, ::-1, :] += tmp #pix

        # params = {"pred0": preds}
        # visualize(**params)

        

        # # # (n, h, w, c) = preds.shape
        # # # (n, c, h, w) = preds.shape
        # for index in range(c):
        #     pix = index + 1
        #     # pred = preds[0, :, :, index:index + 1]
        #     pred = preds
        #     # cv2.imshow("pred0", pred)

        #     # params = {"pred0": pred}
        #     # visualize(**params)

        #     pred[pred > 0.1] = 1
        #     # pred[pred < 0.05] = 0
        #     pred = pred.reshape((dimensions[1], dimensions[0]))
        #     tmp = copyArray[i, ::-1, :] + pred * pix
        #     copyArray[i, ::-1, :] += np.clip(tmp, 0, pix)
        # # # cv2.imshow("pred", copyArray[i, ::-1, :])
        # # # cv2.waitKey(1)

        # params = {"pred": copyArray[i, ::-1, :]}
        # visualize(**params)
    inference_time = time.time()

    copyArray = copyArray.astype(np.uint8)
    # vtk_data = vtk.util.numpy_support.numpy_to_vtk(
    vtk_data = numpy_support.numpy_to_vtk(
        np.ravel(copyArray), dimensions[2], vtk.VTK_UNSIGNED_CHAR
    )
    image = vtk.vtkImageData()
    image.SetDimensions(output.GetDimensions())
    image.SetSpacing(output.GetSpacing())
    image.SetOrigin(output.GetOrigin())
    image.GetPointData().SetScalars(vtk_data)
    output = image
    print(image.GetDimensions())

    contour = vtk.vtkDiscreteMarchingCubes()
    contour.SetInputData(output)
    contour.ComputeNormalsOn()
    # contour.SetValue(0, 1)
    contour.GenerateValues(len(labels), 0, len(labels) + 1)
    contour.Update()
    output = contour.GetOutput()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(output)
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1, 1, 1)
    writer = vtk.vtkSTLWriter()
    writer.SetInputData(output)

    label_name = labels[0]
    uid = image_path.split("\\")[-2]
    stl_save_base_path = os.path.join(entrance["save_base_path"], label_name)
    if not os.path.exists(stl_save_base_path):
        os.makedirs(stl_save_base_path)
    stl_name = os.path.join(
        stl_save_base_path, "{}_{}_{}_{}.stl" .format(
            uid, encoder_name, decoder_name, time.strftime("%m%d_%H%M%S")
        )
    )
    writer.SetFileName(stl_name.encode("GBK"))
    writer.SetFileTypeToBinary()
    writer.Update()

    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderWindow)
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.3)
    interactor.Initialize()
    renderWindow.Render()
    interactor.Start()
    end_time = time.time()
    print("model:", load_model_time - start_time)
    print("inference time:", inference_time - load_model_time)
    print("3D recon time:", end_time - inference_time)
    print("total time:", end_time - start_time)


if __name__ == "__main__":
    entrance = {
        "encoder_name": "efficientnet-b4",
        # "best_model": ".\\output\\pth\\efficientnet-b4_noclip\\efficientnet-b4_epoch_23.pth",
        # "best_model": ".\\output\\pth\\efficientnet-b4\\0316_174217\\efficientnet-b4_epoch_14.pth",
        # "best_model": ".\\output\\pth\\efficientnet-b4\\0315_clip\\efficientnet-b4_epoch_23.pth",
        "best_model": ".\\output\\pth\\efficientnet-b4\\0320_133848\\efficientnet-b4_epoch_24.pth",
        # "best_model": ".\\output\\pth\\efficientnet-b4_MANet\\0321_142242\\efficientnet-b4_epoch_30.pth",
        # "encoder_name": "tu-regnety_040",
        # # "best_model": ".\\output\\pth\\tu-regnety_040_MANet\\0321_173313\\tu-regnety_040_MANet_epoch_35.pth",
        # "best_model": ".\\output\\pth\\tu-regnety_040_MANet\\0323_183024\\tu-regnety_040_MANet_epoch_35.pth",
        # "encoder_name": "resnext101_32x4d",
        # "best_model": ".\\output\\pth\\resnext101_32x4d\\0321_092203\\resnext101_32x4d_epoch_30.pth",
        # "encoder_name": "mobileone_s4",
        # # "best_model": ".\\output\\pth\\mobileone_s4_Unet\\mobileone_s4_epoch_33.pth",
        # "best_model": ".\\output\\pth\\mobileone_s4_Unet\\0322_183333\\mobileone_s4_Unet_epoch_50.pth",
        
        "decoder_name": "MANet", #"Unet", #
        "device": "cuda:0",
        "test_base_path": "D:\\project\\TrueHealth\\20230217_Alg1\\data\\examples\\src_seg\\val",
        "image_path": "D:\\project\\TrueHealth\\20230217_Alg1\\data\\examples\\src_seg\\val\\20170831-000005\\dicom",
        "windowlevel": -600,
        "windowwidth": 2000,
        "middle_patch_size": 512,
        "classes": ["lung"],
        "in_channels": 1,
        "num_workers": 8,  # 多线程加载所需要的线程数目
        "pin_memory": True,  # 数据从CPU->pin_memory—>GPU加速
        "batch_size": 4,
        "save_base_path": "D:\project\TrueHealth\git\segmentation_models.pytorch\output\stl",
    } 

    test_dataset = smp.datasets.SegDataset1(
        base_path=entrance["test_base_path"],
        height=entrance["middle_patch_size"],
        width=entrance["middle_patch_size"],
        channels=entrance["in_channels"],
        windowlevel=entrance["windowlevel"],
        windowwidth=entrance["windowwidth"],
        transform=None,
        # transform=transform,
        status=False
    )
    test(test_dataset, **entrance)
    # generate_stl(**entrance)
    print("Happy End!")