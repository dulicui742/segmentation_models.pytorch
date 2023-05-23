import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import torch
import numpy as np
import vtk
import random
import hiddenlayer as h

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
from affine_matrix import get_affine_matrix


# helper function for data visualization
# def visualize(**images):
    # """PLot images in one row."""
    # n = len(images)
    # plt.figure(figsize=(16, 5))
    # for i, (name, image) in enumerate(images.items()):
    #     plt.subplot(1, n, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.title(' '.join(name.split('_')).title())
    #     plt.imshow(image, cmap=plt.cm.gray)
    # plt.show()
    # plt.pause(1)


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


def load_model(
    encoder_name, 
    decoder_name, 
    model_path, 
    device, 
    in_channels=1, 
    classes=1, 
    output_stride=32,
    decoder_attention_type=None,
):
    ## 实例化
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
        classes=classes,
        **kwargs,
    ).to(device)

    ## load model parameters
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=False)
    return model


def vis_model_graph(model, **entrance):
    batch_size = entrance["batch_size"]
    height = entrance["middle_patch_size"]
    width = entrance["middle_patch_size"]
    device = entrance["device"]
    encoder_name = entrance["encoder_name"]
    decoder_name = entrance["decoder_name"]

    x = torch.zeros([batch_size, 1, height, width]).float().to(device)

    model_decoder = model.decoder
    model_encoder = model.encoder

    vis_graph = h.build_graph(model_encoder, x)   # 获取绘制图像的对象
    vis_graph.theme = h.graph.THEMES["blue"].copy()     # 指定主题颜色
    
    graph_base_path = os.path.join(".\\output", "graph")
    if not os.path.exists(graph_base_path):
        os.makedirs(graph_base_path)
    graph_name = os.path.join(graph_base_path, "{}_{}_encoder.png" .format(encoder_name, decoder_name))
    vis_graph.save(graph_name)   # 保存图像的路径


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
    decoder_name = entrance["decoder_name"]
    device = torch.device(entrance["device"] if torch.cuda.is_available() else "cpu")
    model_path = entrance["best_model"]
    classes = entrance["classes"]
    num_classes = len(classes)
    output_stride = entrance["output_stride"]
    in_channels = entrance["in_channels"]
    model = load_model(
        encoder_name, 
        decoder_name,
        model_path, 
        device, 
        in_channels=in_channels, 
        classes=num_classes, 
        output_stride=output_stride
    )

    ### re-parameterizable
    if  "mobileone" in encoder_name:
        print("~~~~~~~~~~~Re-parameter starts!~~~~~~~~~~")
        from segmentation_models_pytorch.encoders.mobileone import reparameterize_model
        best_model = reparameterize_model(model) ## 
    else:
        best_model = model

    model_time = time.time()

    # if entrance["vis_graph"]:
    #     vis_model_graph(model, **entrance)
    
    # evaluate model on test set
    random.seed(1)
    index = random.sample(range(0, len(test_dataset)), 20)
    print(index)
    # index = range(len(test_dataset))
    for i in index:
        n = i
        image, gt_mask = test_dataset[n]
        # gt_mask = gt_mask.squeeze()
        
        x_tensor = torch.from_numpy(image).to(device).unsqueeze(0).float()
        pr_mask = best_model.predict(x_tensor)

        pr_mask = torch.sigmoid(pr_mask)
        pr_mask = (pr_mask.squeeze(0).cpu().numpy().round())

        # image = image.squeeze().astype(np.uint8)
        image = image.astype(np.uint8) 
        params = []
        for i in range(len(classes)):
            params.append({
                "image": image[i,:,:], 
                "gt_{}" .format(classes[i]): gt_mask[i,:,:], 
                "pred_{}" .format(classes[i]): pr_mask[i,:,:]})
        visualize(params)

    print("model time:", model_time - start_time)
    print("inference time:", time.time() - start_time)


def generate_stl(**entrance):
    start_time = time.time()
    encoder_name = entrance["encoder_name"]
    decoder_name = entrance["decoder_name"]
    # image_path = entrance["image_path"]
    image_base_path = entrance["image_base_path"]
    device = torch.device(entrance["device"] if torch.cuda.is_available() else "cpu")
    windowlevel = entrance["windowlevel"]
    windowwidth = entrance["windowwidth"]
    best_model_path = entrance["best_model"]
    labels = entrance["classes"]
    num_classes = len(entrance["classes"])
    output_stride = entrance["output_stride"]
    stragety = entrance["stragety"]
    sigmoid_threshold = entrance["sigmoid_threshold"]
    timestamp = "_".join(best_model_path.split("\\")[-1].split("_")[-3:-1])
    print("ts:", timestamp)

    # best_model = load_model(encoder_name, best_model_path, device)
    best_model = load_model(
        encoder_name, 
        decoder_name,
        best_model_path, 
        device, 
        in_channels=1, 
        classes=num_classes, 
        output_stride=output_stride
    )
    load_model_time = time.time()

    time_dict = {}
    uids = os.listdir(image_base_path)
    for uid in uids:
        start_infernce = time.time()
        if "txt" in uid or "xlsx" in uid: ## except readme
            continue
        
        # if uid in ["PA1", "PA13", "PA11", "PA12", "PA10", "PA2"]:
        # if uid in ["PA2"]:
        #     print("skip: ", uid)
        #     continue

        # if uid not in ["35", "37", "43", "81", "82", "83", "84", "85", "86", "87", "88", "89", "52", "53", "62"]: # 
        #     continue

        print("\n------------------------------")
        print("dealing with: {}" .format(uid))
        time_dict[uid] = {}

        # import pdb; pdb.set_trace()
        image_path = os.path.join(image_base_path, uid, "dicom")
        dicomreader = vtk.vtkDICOMImageReader()
        dicomreader.SetDirectoryName(image_path)
        dicomreader.Update()
        output = dicomreader.GetOutput()
        dimensions = output.GetDimensions()
        print("dimension:", dimensions)

        dicomArray = numpy_support.vtk_to_numpy(output.GetPointData().GetScalars())
        dicomArray = dicomArray.reshape(dimensions[::-1]).astype(np.float32)
        copyArray = dicomArray * 0

        for i in range(dimensions[2]):
            img = dicomArray[i, ::-1, :] 
            # ## method1
            # img = ((img - windowlevel) / windowwidth + 0.5) * 255.0

            # ## method2 
            img = (img - windowlevel) / windowwidth + 0.5
            img = np.clip(img, 0, 1) * 255

            # cv2.imshow("img", img/255)
            # params = {"image": img}
            # visualize(**params)
            # img = img.reshape((1, dimensions[1], dimensions[0]))
            img = img[np.newaxis,:,:]
            x_tensor = torch.from_numpy(img).to(device).unsqueeze(0).float()
            preds = best_model.predict(x_tensor)  # 预测图
            preds = torch.sigmoid(preds)
            (n, c, h, w) = preds.shape
            # preds = (preds.squeeze().cpu().numpy().round()) ## shape: 512 * 512
            preds = (preds.squeeze().cpu().numpy())
            preds[preds > sigmoid_threshold] = 1
            # preds[preds > 0.1] = 1
            # params = {"pred0": preds}
            # visualize(**params)

            # tmp = copyArray[i, ::-1, :] + preds * 1 #pix
            # copyArray[i, ::-1, :] += np.clip(tmp, 0, 1) #pix
            copyArray[i, ::-1, :] = np.clip(preds, 0, 1)

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

        matrix = get_affine_matrix(image_path)
        print("-------------\n", matrix)

        def Array2vtkTransform(arr):
            T = vtk.vtkTransform()
            matrix = vtk.vtkMatrix4x4()
            for i in range(0, 4):
                for j in range(0, 4):
                    matrix.SetElement(i, j, arr[i, j])
            T.SetMatrix(matrix)
            return T

        # transform = vtk.vtkTransform()
        # transform.SetMatrix(matrix)
        transform = Array2vtkTransform(matrix)
        transformPolyDataFilter = vtk.vtkTransformPolyDataFilter()
        transformPolyDataFilter.SetInputData(output)
        transformPolyDataFilter.SetTransform(transform)
        transformPolyDataFilter.Update()
        output = transformPolyDataFilter.GetOutput()

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
        epoch = int(best_model_path.split("_")[-1].split(".")[0])
        stl_save_base_path = os.path.join(
            entrance["save_base_path"],  
            label_name,
            "{}_{}" .format(encoder_name, decoder_name),
            stragety,
            timestamp,
            "epoch{}_wl{}_ww{}_sigmoid{}" .format(epoch, windowlevel, windowwidth, str(sigmoid_threshold)),
        )
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
        # print("inference time:", inference_time - load_model_time)
        print("inference time:", inference_time - start_infernce)
        print("3D recon time:", end_time - inference_time)
        # print("total time:", end_time - start_time)
        print("total time:", end_time - start_infernce + (load_model_time - start_time))

        time_dict[uid]["slice Num"] = dimensions[2]
        time_dict[uid]["Load Model"] = round(load_model_time - start_time, 3)
        time_dict[uid]["Inference"] = round(inference_time - start_infernce, 3)
        time_dict[uid]["3D Recon"] = round(end_time - inference_time, 3)
        time_dict[uid]["Total"] = round(end_time - start_infernce + (load_model_time - start_time), 3)
    ### print table
    print_test_time(encoder_name, time_dict)


def print_test_time(encoder_name, time_dict):
    from prettytable import PrettyTable
    test_samples = list(time_dict.keys())
    x = PrettyTable()
    x.title = "Test time of {}" .format(encoder_name)
    x.field_names = ["Test sample"] + list(time_dict[test_samples[0]].keys())

    for sp in test_samples:
        values = list(time_dict[sp].values())
        tmp = []
        tmp.append(sp)
        tmp.extend(values)
        x.add_rows([
            tmp
        ])
    print(x)


if __name__ == "__main__":
    entrance = {
        # "encoder_name": "efficientnet-b4",
        # # # # # "best_model": ".\\output\\pth\\efficientnet-b4_noclip\\efficientnet-b4_epoch_23.pth",
        # # # # # "best_model": ".\\output\\pth\\efficientnet-b4\\0316_174217\\efficientnet-b4_epoch_14.pth",
        # "best_model": ".\\output\\pth\\efficientnet-b4\\0315_clip\\efficientnet-b4_epoch_23.pth",
        # # "best_model": ".\\output\\pth\\efficientnet-b4_Unet_normal-rotated\\0404_183653\\efficientnet-b4_Unet_normal-rotated_epoch_20.pth",
        # # "best_model": ".\\output\\pth\\efficientnet-b4\\0320_133848\\efficientnet-b4_epoch_24.pth",
        # "best_model": ".\\output\\pth\\efficientnet-b4_MANet\\0321_142242\\efficientnet-b4_epoch_30.pth",
        # "best_model": "D:\share\efficientnet-b4_Unet_clip_rotated_epoch_46.pth",
         
        "encoder_name": "stdc2",
        # # # # # "best_model": ".\\output\\pth\\stdc2_Unet_clip-rotated\\0331_144655\\stdc2_Unet_clip-rotated_epoch_11.pth",
        # "best_model": "D:\\share\\stdc\\stdc2_Unet_clip-rotated_epoch_50.pth",
        # "best_model": "D:\share\stdc\pth\stdc2_unet\stdc2_Unet_clip-rotated_epoch_36.pth",
        # # # "best_model": ".\\output\\pth\\stdc2_Unet_clip-rotated\\0406_183556\\stdc2_Unet_clip-rotated_epoch_66.pth",
        # # "best_model": ".\\output\pth\stdc2_Unet\clip-rotated-32x-customLR1\\0413_101447\stdc2_Unet_clip-rotated-32x-customLR1_epoch_60.pth",
        # # "best_model": "D:\share\stdc\stdc2_Unet_clip-rotated_8x_epoch_10.pth",
        # "best_model": "D:\share\stdc\pth\stdc2_Unet_clip-rotated-8x-OneCycleLR_epoch_5.pth",
        # "best_model": "D:\share\stdc\pth\stdc2_Unet_clip-rotated-16x-customLR1_epoch_44.pth",
        "best_model": ".\output\pth\stdc2_Unet\clip-rotated-16x-customLR1\\0418_093532\stdc2_Unet_clip-rotated-16x-customLR1_epoch_28.pth",
        # "best_model": "D:\share\stdc\pth\stdc2_MANet_noclip-rotated_epoch_22.pth",
        # "best_model": ".\output\pth\stdc2_Unet\clip-rotated-32x-customLR1\\0423_110239\stdc2_Unet_clip-rotated-32x-customLR1_epoch_56.pth",
        # "best_model": "D:\share\stdc\pth\stdc2_MANet_clip-rotated_epoch_21.pth",
        # "best_model": "D:\share\AttUnet\pth\stdc2_AttentionUnet_clip-rotated-focal-OneCycleLR-32x-OneCycleLR_epoch_12.pth",
        # "best_model": ".\output\pth\stdc2_Unet\clip-rotated-32x-customLR1\\0426_144612\stdc2_Unet_clip-rotated-32x-customLR1_epoch_23.pth",
        # "best_model": ".\output\pth\stdc2_Unet\clip-rotated-32x-customLR1\\0427_132240\stdc2_Unet_clip-rotated-32x-customLR1_epoch_52.pth",
        # "best_model": ".\output\pth\stdc2_Unet\clip-rotated-32x-customLR1\\0427_164303\stdc2_Unet_clip-rotated-32x-customLR1_epoch_57.pth",
        # "best_model": "D:\share\stdc\pth\\16x\stdc2_Unet_clip-rotated-16x-customLR1_epoch_19.pth",
        # "best_model": "D:\share\stdc\ck\stdc2_MANet_clip-rotated-focalloss_epoch_12.pth",
        # "best_model": "D:\share\stdc\pth\stdc2_unet\stdc2_Unet_clip-rotated-32x-customLR1_epoch_28.pth", #zhiqiguan 18,24,28
        # "best_model": "D:\share\stdc\pth\stdc2_unet\\bronchial\epoch_0506_172018_30.pth",
        # "best_model": "D:\share\stdc\pth\stdc2_unet\\bronchial\stdc2_Unet_clip-rotated-32x-customLR1_epoch_0504_143917_46.pth",
        # "best_model": "D:\share\stdc\pth\stdc2_unet\PulmonaryVessels\clip-rotated-bce-adam-customLR1-32x-wd0\\0509_153825\epoch_0509_153825_20",
        # "best_model": ".\output\stdc2_Unet\clip-rotated-class3-customLR1-32x\pth\\0511_132746\epoch_0511_132746_40",
        # "best_model": ".\output\stdc2_Unet\clip-rotated-class3-customLR1-32x\pth\\0511_162839\epoch_0511_162839_21",
        # "best_model": ".\output\stdc2_Unet\clip-rotated-class3-customLR1-32x\pth\\0511_171104\epoch_0511_171104_29",
        # "best_model": "D:\share\stdc\pth\stdc2_unet\\bronchial\clip-rotated-bce-adam-customLR1-32x-wd0\epoch_0510_141835_11", ##zhiqiguan

        "decoder_name": "Unet", #"AttentionUnet", #"MANet", #
        "device": "cuda:0",
        "test_base_path": "D:\\project\\TrueHealth\\20230217_Alg1\\data\\examples\\src_seg\\val",
        "image_path": "D:\project\TrueHealth\\20230217_Alg1\data\examples\src_seg\zhiqiguan_test\PA1\dicom",
        "image_base_path": "D:\project\TrueHealth\\20230217_Alg1\data\examples\src_seg\zhiqiguan_test",
        # "image_base_path": "\\192.168.1.99\Data\data_sphere_test",
        
        # "windowlevel": -600,
        # "windowwidth": 2000,

        "windowlevel": -850,
        "windowwidth": 310,

        # "windowlevel": 0,
        # "windowwidth": 2000,

        # "windowlevel": -75,
        # "windowwidth": 800,

        # "windowlevel": [-600, 135, 50, -850],  ## lung, skin, heart, zhiqiguan
        # "windowwidth": [2000, 385, 500, 310],

        "middle_patch_size": 512,
        "classes": ["zhiqiguan"],# , ["lung", "skin", "heart"], #["lung"], #["lung", "skin", "heart", "zhiqiguan"],#["PulmonaryVessels"], # 
        "in_channels": 1,
        "num_workers": 8,  # 多线程加载所需要的线程数目
        "pin_memory": True,  # 数据从CPU->pin_memory—>GPU加速
        "batch_size": 4,
        "save_base_path": "D:\project\TrueHealth\git\segmentation_models.pytorch\output\stl",

        "output_stride": 16,
        # "stragety": "clip-rotated-32x-customLR-wd1e5", #"clip-rotated-32x-focal", #
        # "stragety": "clip-rotated-bce-adam-customLR1-32x-wd10-5",
        # "stragety": "clip-rotated-bce-adam-customLR1-32x-wd0",
        "stragety": "clip-rotated-16x-customLR1",
        # "stragety": "clip-rotated-class3",
        "sigmoid_threshold": 0.5,
        # "vis_graph": True,
        "is_multilabels": False
    } 

    test_dataset = smp.datasets.SegDataset1(
        base_path=entrance["test_base_path"],
        stl_names=entrance["classes"],
        height=entrance["middle_patch_size"],
        width=entrance["middle_patch_size"],
        channels=entrance["in_channels"],
        windowlevel=entrance["windowlevel"],
        windowwidth=entrance["windowwidth"],
        transform=None,
        # transform=transform,
        status=False,
        is_multilabels=entrance["is_multilabels"],
    )
    # test(test_dataset, **entrance)
    generate_stl(**entrance)
    print("Happy End!")