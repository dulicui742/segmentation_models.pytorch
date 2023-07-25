
import os
import numpy as np
import cv2
import random
import time
import torch
import numpy as np
import vtk
import itk
import matplotlib.pyplot as plt
import pyclipper
import albumentations as A

from vtk.util import numpy_support
from vtk import vtkMatrix4x4

from itertools import chain


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
from segmentation_models_pytorch.utils.losses import(
    JaccardLoss,
    DiceLoss,
    BCELoss
)
from segmentation_models_pytorch.utils.metrics import(
    IoU, Fscore, Precision, Recall, Accuracy
)
from affine_matrix import (
    get_affine_matrix,
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
    plt.pause(2)


def load_model(
    encoder_name, 
    decoder_name, 
    model_path, 
    device, 
    in_channels=1, 
    classes=1, 
    encoder_depth=5,
    decoder_channels=[256, 128, 64, 32, 16],
    output_stride=32,
    decoder_attention_type=None,
):
    ## 实例化
    if decoder_name == "Unet" or decoder_name == "AttentionUnet":
        kwargs = {
        "output_stride": output_stride, 
        # "decoder_attention_type": decoder_attention_type, 
    }
    else:
        kwargs = {}

    model = smp.create_model(
        arch=decoder_name,
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        encoder_depth=encoder_depth,
        decoder_channels=decoder_channels,
        classes=classes,
        **kwargs,
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
    decoder_name = entrance["decoder_name"]
    device = torch.device(entrance["device"] if torch.cuda.is_available() else "cpu")
    model_path = entrance["best_model"]
    labels = entrance["classes"]
    num_classes = len(labels)
    output_stride = entrance["output_stride"]
    decoder_channels = entrance["decoder_channels"]
    encoder_depth = entrance["encoder_depth"]

    # model = load_model(
    #     encoder_name, decoder_name, model_path, device, in_channels=1, classes=len(labels),
    #     decoder_attention_type=entrance["decoder_attention_type"],
    # )
    model = load_model(
        encoder_name, 
        decoder_name,
        model_path, 
        device, 
        in_channels=in_channels, 
        encoder_depth=encoder_depth,
        decoder_channels=decoder_channels,
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
 
    # evaluate model on test set
    random.seed(2)
    index = random.sample(range(0, len(test_dataset)), 40)
    # index = range(len(test_dataset)//2 - 8, len(test_dataset)//2 + 18)
    for i in index:
        n = i
        image, gt_mask = test_dataset[n]
        
        gt_mask = gt_mask.squeeze()
        
        x_tensor = torch.from_numpy(image).to(device).unsqueeze(0).float()
        pr_mask = best_model.predict(x_tensor)

        pr_mask = torch.sigmoid(pr_mask)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        # pr_mask[pr_mask>0.5]=1

        image = image.squeeze().astype(np.uint8) 
        params = {"image": image, "ground_truth_mask": gt_mask, "predicted_mask": pr_mask}
        visualize(**params)

    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("model time:", model_time - start_time)
    print("inference time:", time.time() - start_time)


def foreground(img, margin_extend=40, margin_inside=-30, idx=0):
    # print("========================")
    # print(margin_extend, margin_inside)
    # print("========================")
    img_bk = img.copy()
    img = img.astype(np.uint8)
    gaussian = cv2.medianBlur(img, 3)
    # ret, edges = cv2.threshold(gaussian, 127, 255, cv2.THRESH_BINARY)
    ret, edges =  cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    contours, hierarchy = cv2.findContours(
        # edges,
        opening, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    ) #cv2.RETR_TREE

    area = []
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    max_idx = np.argmax(np.array(area))
    
    # ### ===============minEnclosingCircle==============
    # (x, y), radius = cv2.minEnclosingCircle(contours[max_idx])
    # center = (int(x), int(y))
    # radius = int(radius)

    # new = np.zeros(img.shape, dtype=np.uint8)
    # newrgb = cv2.cvtColor(new, cv2.COLOR_GRAY2BGR)
    # imgrgb =  cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # # cv2.circle(imgrgb, center, radius, (0, 255, 0), 1)
    # # cv2.circle(newrgb, center, radius, (0, 255, 0), -1)
    # cv2.circle(newrgb, center, radius+10, (0, 255, 0), -1) ##向外扩展
    # mask = cv2.cvtColor(newrgb, cv2.COLOR_BGR2GRAY)
    # mask[mask > 0] = 1
    # edges[edges > 0] = 1
    # # mask *= edges ## 二值化和外接圆共同确定前景
    # img_bk = img_bk * mask ## 使用最大轮廓的外接圆做mask
    # ### ===============minEnclosingCircle==============

    outmask = np.zeros(img.shape, dtype=np.uint8)
    outmaskrgb = cv2.cvtColor(outmask, cv2.COLOR_GRAY2BGR)
    inmask = np.zeros(img.shape, dtype=np.uint8)
    inmaskrgb = cv2.cvtColor(inmask, cv2.COLOR_GRAY2BGR)

    contour_extend = equidistant_zoom_contour(contours[max_idx], margin=margin_extend)
    contour_inside = equidistant_zoom_contour(contours[max_idx], margin=margin_inside)
    maskout = cv2.drawContours(outmaskrgb, [contour_extend], -1, 1, cv2.FILLED)
    maskin = cv2.drawContours(inmaskrgb, [contour_inside], -1, 1, cv2.FILLED)

    mask = maskout[:,:,0]
    mask_ring = maskout[:,:,0] - maskin[:,:,0]
    # img_fg = img_bk * mask_ring  ## 使用最大轮廓mask
    
    opening[opening > 0] = 1
    # img_fg = img_bk * mask
    # params = {"ori-image": img_bk, "binary": edges, "mask": mask_ring, "fground": img_ring, }
    # visualize(**params)
    # img_ring[img_ring == 0] = 127

    img_fg = img_bk * mask_ring #* opening  ## 使用最大轮廓mask ##使用opening之后，FP变多
    return img_fg, mask_ring


def equidistant_zoom_contour(contour, margin):
    """
    等距离缩放多边形轮廓点
    :param contour: 一个图形的轮廓格式[[[x1, x2]],...],shape是(-1, 1, 2)
    :param margin: 轮廓外扩的像素距离，margin正数是外扩，负数是缩小
    :return: 外扩后的轮廓点
    """
    pco = pyclipper.PyclipperOffset()
    ##### 参数限制，默认成2这里设置大一些，主要是用于多边形的尖角是否用圆角代替
    pco.MiterLimit = 10
    contour = contour[:, 0, :]
    pco.AddPath(contour, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
    solution = pco.Execute(margin)

    ### dulicui 
    ### solution's length is not always 1, so get the longest first!!!
    solution = max(solution, key=lambda k: len(k)) ## 获取长度最大的solution
    solution = np.array(solution).reshape(-1, 1, 2).astype(int)
    return solution


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
    stragety = entrance["stragety"]
    sigmoid_threshold = entrance["sigmoid_threshold"]
    output_stride = entrance["output_stride"]
    num_classes = len(entrance["classes"])
    decoder_channels = entrance["decoder_channels"]
    encoder_depth = entrance["encoder_depth"]

    timestamp = best_model_path.split("/")[-2]

    best_model = load_model(
        encoder_name, 
        decoder_name,
        best_model_path, 
        device, 
        in_channels=1, 
        encoder_depth=encoder_depth,
        decoder_channels=decoder_channels,
        classes=num_classes, 
        output_stride=output_stride
    )
    load_model_time = time.time()

    time_dict = {}
    uids = os.listdir(image_base_path)
    uids = [i for i in uids if "xls" not in i]
    uids.sort(key=lambda x: int(x))
    print(uids)
    for uid in uids:
        start_infernce = time.time()
        if "txt" in uid or "xlsx" in uid: ## except readme
            continue
        
        print("\n------------------------------")
        print("dealing with: {}" .format(uid))
        time_dict[uid] = {}

        image_path = os.path.join(image_base_path, uid, "dicom")
        # image_path = os.path.join(image_base_path, uid,)
        dicomreader = vtk.vtkDICOMImageReader()
        dicomreader.SetDirectoryName(image_path)
        dicomreader.Update()
        output = dicomreader.GetOutput()
        dimensions = output.GetDimensions()
        slice_spacing = output.GetSpacing()
        print("dimension:", dimensions, "spacing:", slice_spacing)

        dicomArray = numpy_support.vtk_to_numpy(output.GetPointData().GetScalars())
        dicomArray = dicomArray.reshape(dimensions[::-1]).astype(np.float32)
        # dicomArray = dicomArray.reshape(dimensions[2], dimensions[0], dimensions[1]).astype(np.float32)
        copyArray = dicomArray * 0

        slice_fg_pixel_nums = []
        for i in range(dimensions[2]):
            img = dicomArray[i, ::-1, :]  # slice

            if entrance["aug_name"] == "noclip-rotated":
                img = ((img - windowlevel) / windowwidth + 0.5) * 255
            elif entrance["aug_name"] == "clip-rotated":
                ## for test
                img = ((img - windowlevel) / windowwidth + 0.5) 
                img = np.clip(img, 0, 1) * 255
            else:
                print("Error!!!")
                break
            
            # cv2.imshow("img", img/255)
            # params = {"image": img}
            # visualize(**params)
            # img = img.reshape((1, dimensions[1], dimensions[0]))
            # img = img.reshape((1, dimensions[0], dimensions[1]))

            ### preprocess
            img, _ = foreground(img, idx=i)
            # img = (img/255 - 0.5) * windowwidth + windowlevel
            # copyArray[i, ::-1, :] = img

            img = img[np.newaxis, :, :]

            x_tensor = torch.from_numpy(img).to(device).unsqueeze(0).float()
            preds = best_model.predict(x_tensor)  # 预测图
            (n, c, h, w) = preds.shape

            for index in range(c):
                # pix = index + 1
                # pred = preds[0, :, :, index:index + 1]
                pred = preds[:, index, :, :] 
                pred = torch.sigmoid(pred)
                pred = pred.squeeze().cpu().numpy()
          
                # if np.max(pred) > sigmoid_threshold:
                #     num = np.argwhere(pred > sigmoid_threshold).shape[0]
                #     print(i, np.max(pred), np.min(pred), num)

                # pred = pred * mask ## 用mask约束
                pred[pred > sigmoid_threshold] = 1
                pred[pred <= sigmoid_threshold] = 0

                ###=================================
                cur_num = np.argwhere(pred > sigmoid_threshold).shape[0]
                slice_fg_pixel_nums.append(cur_num)
                ###=================================

                if np.max(pred) > sigmoid_threshold:
                    num = np.argwhere(pred > sigmoid_threshold).shape[0]
                    print(i, np.max(pred), np.min(pred), num)

                # pred = pred.astype(np.uint8)
                # contours, hierarchy = cv2.findContours(
                #     pred, 
                #     cv2.RETR_EXTERNAL, 
                #     cv2.CHAIN_APPROX_SIMPLE
                # ) #cv2.RETR_TREE

                # # area = []
                # for k in range(len(contours)):
                #     area = cv2.contourArea(contours[k])
                #     if area > 250 or area < 8:
                #         num1 = np.argwhere(pred > sigmoid_threshold).shape[0]
                #         cv2.drawContours(pred, contours, k, 0, cv2.FILLED)
                #         num2 = np.argwhere(pred > sigmoid_threshold).shape[0]
                #         print(i, num1, num2)

                copyArray[i, ::-1, :] = np.clip(pred, 0, 1)
            # params = {"pred": copyArray[i, ::-1, :]}
            # visualize(**params)

        ##============remove non continus =================
        def continusFind(numlist):
            continus_res = []
            l1 = []
            for x in sorted(set(numlist)):
                l1.append(x)
                if x+1 not in numlist:
                    if len(l1) != 1:
                        continus_res.append(l1)
                    l1 = []
            return continus_res
        
        non_zero_idx = np.nonzero(np.array(slice_fg_pixel_nums))
        non_zero_idx = non_zero_idx[0].tolist()
        cres = continusFind(non_zero_idx)
        
        fg_res = []
        if slice_spacing[-1] < 2:
            slice_nums = 7
        else:
            slice_nums = 4
        for cc in cres:
            if len(cc) > slice_nums: # and max(cc) > 100: # 4个以上的连续才且中间层的pixel数量大于100认为是sphere
                fg_res.append(cc)
        print(fg_res)
        fg_idx = list(chain(*fg_res))
        fgArray = copyArray * 0
        for idx in fg_idx:
            fgArray[idx,:,:] = copyArray[idx,:,:]

        ### save
        # cv2.imwrite(f"./pred_80_before.png", copyArray[80,::-1,:]*255)
        # cv2.imwrite(f"./pred_80_after.png", fgArray[80,::-1,:]*255)

        # cv2.imwrite(f"./pred_82_before.png", copyArray[82,::-1,:]*255)
        # cv2.imwrite(f"./pred_82_after.png", fgArray[82,::-1,:]*255)

        ##============remove non continus =================
        inference_time = time.time()
        # copyArray = copyArray.astype(np.uint8)
        # vtk_data = numpy_support.numpy_to_vtk(
        #     np.ravel(copyArray), dimensions[2], vtk.VTK_UNSIGNED_CHAR
        # )

        fgArray = fgArray.astype(np.uint8)
        vtk_data = numpy_support.numpy_to_vtk(
            np.ravel(fgArray), dimensions[2], vtk.VTK_UNSIGNED_CHAR
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
        print("-------------")
        print(matrix)

        def Array2vtkTransform(arr):
            T = vtk.vtkTransform()
            matrix = vtk.vtkMatrix4x4()
            for i in range(0, 4):
                for j in range(0, 4):
                    matrix.SetElement(i, j, arr[i, j])
            T.SetMatrix(matrix)
            # T.SetUserMatrix(matrix)
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
        # interactor.Initialize()
        # renderWindow.Render()
        # interactor.Start()
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
        # break
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
        # # # # # # "best_model": "/home/th/Data/dulicui/share/efficientnet-b4_MANet_epoch_16.pth", #21， 41， 56,63, 66, 
        # # # # # # "best_model": "/home/th/Data/dulicui/share/efficientnet-b4_MANet_clip-rotated_epoch_69.pth", #71
        # # # # # # "best_model": "/home/th/Data/dulicui/share/efficientnet-b4_epoch_23.pth", #21 0321-151721
        # # # # "best_model": "./output/pth/efficientnet-b4_Unet_clip_rotated/0329_100345/efficientnet-b4_Unet_clip_rotated_epoch_46.pth", #26(x), 46(ok)
        # # # # "best_model": "./output/pth/efficientnet-b4_Unet/0324_182306/efficientnet-b4_Unet_epoch_18.pth",
        # # "best_model": "./output/pth/efficientnet-b4_Unet_clip_rotated-focalloss/0421_095419/efficientnet-b4_Unet_clip_rotated-focalloss_epoch_27.pth",
        # "best_model": "./output/pth/efficientnet-b4_Unet_clip_rotated-focalloss/0421_095419/efficientnet-b4_Unet_clip_rotated-focalloss_epoch_20.pth",

        "encoder_name": "stdc2",
        # "encoder_name": "stdc1",
        ## bronchial
        # "best_model": "./output/pth/Bronchial/Stdc2_Unet/clip-rotated-bce-adam-customLR1-32x-wd0/0331_163802/stdc2_Unet_clip-rotated_epoch_50.pth", #Bronchial
        # "best_model": "./output/pth/Bronchial/Stdc2_MANet/clip-rotated-bce-adam-customLR1-32x-wd0/0421_164543/stdc2_MANet_clip-rotated_epoch_18.pth",
        # "best_model": "./output/pth/Bronchial/Stdc2_Unet/clip-rotated-bce-adam-customLR1-32x-wd0.0001/0506_172018/epoch_0506_172018_30.pth",
        # "best_model": "./output/pth/Bronchial/Stdc2_Unet/clip-rotated-16x-customLR1/0418_093532/stdc2_Unet_clip-rotated-16x-customLR1_epoch_28.pth",
        
        # # "best_model": "/home/th/Data/dulicui/share/stdc2_Unet_clip-rotated-scse_epoch_60.pth",
        # # "best_model": "./output/pth/stdc2_Unet_clip_rotated/0407_111627/stdc2_Unet_clip_rotated_epoch_15.pth", ##10 ##16 Sphere 24, 12, 18(ok), 23
        # "best_model": "./output/pth/stdc2_Unet_clip_rotated-focalloss/0418_150638/stdc2_Unet_clip_rotated-focalloss_epoch_18.pth",
        # # "best_model": "./output/pth/stdc2_Unet_clip_rotated-focalloss/0419_181332/stdc2_Unet_clip_rotated-focalloss_epoch_25.pth", #sgd
        # "best_model": "./output/pth/stdc2_Unet_clip_rotated/0410_183547/stdc2_Unet_clip_rotated_epoch_47.pth", ## Lung 4,13，21, 23
        # "best_model": "./output/pth/stdc2_MANet_clip_rotated-focal/0425_154434/stdc2_MANet_clip_rotated-focal_epoch_16.pth", ##sphere
        # "best_model": "./output/pth/stdc2_MANet_clip-rotated-bce/0426_133144/stdc2_MANet_clip-rotated-bce_epoch_31.pth",
        # "best_model": "./output/pth/stdc2_MANet_clip-rotated-dice-bce/0504_150254/stdc2_MANet_clip-rotated-sum_epoch_7.pth", ##sphere sum
        # "best_model": "./output/pth/stdc2_Unet_clip-rotated-dice-focal/0505_093416/stdc2_Unet_clip-rotated-dice-focal_epoch_19.pth",
        # "best_model": "./output/pth/stdc2_Unet_clip-rotated-dice-focal/0506_144122/stdc2_Unet_clip-rotated-dice-focal_epoch_20.pth",
        # "best_model": "./output/pth/stdc2_Unet_clip-rotated-dice-focal/0508_112817/stdc2_Unet_clip-rotated-dice-focal_epoch_21.pth",
        # "best_model": "./output/pth/LungLobe/stdc2_Unet/clip-rotated-bce-adam-customLR1-32x-wd0/0509_141002/epoch_0509_141002_3",
        # "best_model": "./output/pth/PulmonaryVessels/stdc2_Unet/clip-rotated-bce-adam-customLR1-32x-wd0/0509_153825/epoch_0509_153825_35",
        # 

        ## Sphere
        # "best_model": "./output/pth/stdc2_Unet_clip_rotated-focalloss/0418_150638/stdc2_Unet_clip_rotated-focalloss_epoch_13.pth",
        "best_model": "./output/pth/Sphere/stdc2_Unet/clip-rotated-focal-adam-customLR1-16x-wd0.0001/0526_150357/epoch_0526_150357_12",
        # "best_model": "./output/pth/Sphere/stdc1_Unet/clip-rotated-focal-adam-customLR1-16x-wd0.0001/0531_102350/epoch_0531_102350_9",
        # "best_model": "./output/pth/Sphere/stdc2_Unet/clip-rotated-focal-adam-customLR1-16x-wd0/0602_183634/epoch_0602_183634_24",
        # "best_model": "./output/pth/Sphere/stdc2_Unet/clip-rotated-tverskyfocal-adam-customLR1-16x-wd0.0001/0608_171125/epoch_0608_171125_4",
        # "best_model": "./output/pth/Sphere/stdc2_Unet/clip-rotated-focal-adam-customLR1-16x-wd0.0001/0612_173112/epoch_0612_173112_12",
        # "best_model": "./output/pth/Sphere/stdc2_AttentionUnet/clip-rotated-focal-adam-customLR1-16x-wd0.0001/0613_161238/epoch_0613_161238_13",
        # "best_model": "./output/pth/Sphere/stdc2_AttentionUnet/clip-rotated-focal-adam-customLR1-16x-wd0.0001/0616_131006/epoch_0616_131006_12",



        ## Heart
        # "best_model": "./output/pth/Heart/stdc2_Unet/clip-rotated-bce-adam-customLR1-32x-wd0.0001/0518_182334/epoch_0518_182334_48",
        
        ## Bone
        # "best_model": "./output/pth/Bone/stdc2_Unet/clip-rotated-bce-adam-customLR1-32x-wd0.0001/0518_184607/epoch_0518_184607_49.pth",
        # "best_model": "./output/pth/Bone/stdc2_Unet/clip-rotated-focal-adam-customLR1-32x-wd0.0001/0526_143659/epoch_0526_143659_31.pth",
        # "best_model": "./output/pth/Bone/stdc2_Unet/noclip-rotated-bce-adam-customLR1-32x-wd0.0001/0602_150928/epoch_0602_150928_42.pth",
        # "best_model": "./output/pth/Bone/stdc2_Unet/noclip-rotated-bce-adam-customLR1-32x-wd0/0605_132917/epoch_0605_132917_46.pth",
        # "best_model": "./output/pth/Bone/stdc2_Unet/noclip-rotated-bce-adam-customLR1-32x-wd0.0001/0608_104517/epoch_0608_104517_32.pth",


        ## Total
        # "best_model": "./output/pth/TotalPulmonaryVessels/stdc2_Unet/clip-rotated-bce-adam-customLR1-32x-wd0/0516_171726/epoch_0516_171726_22", #13
        # "best_model": "./output/pth/TotalPulmonaryVessels/stdc2_Unet/clip-rotated-bce-adam-customLR1-32x-wd0/0519_182359/epoch_0519_182359_33",
        # "best_model": "./output/pth/TotalPulmonaryVessels/stdc2_Unet/clip-rotated-bce-adam-customLR1-32x-wd0.0001/0523_121225/epoch_0523_121225_56",
        # "best_model": "./output/pth/TotalPulmonaryVessels/stdc2_Unet/clip-rotated-bce-adam-customLR1-32x-wd0.0005/0525_132949/epoch_0525_132949_33",
        # "best_model": "./output/pth/TotalPulmonaryVessels/stdc2_Unet/clip-rotated-bce-adam-customLR1-32x-wd0.0001/0526_145753/epoch_0526_145753_16",
        # "best_model": "./output/pth/TotalPulmonaryVessels/stdc2_Unet/clip-rotated-bce-adam-customLR1-16x-wd0.0001/0529_134445/epoch_0529_134445_48",
        # "best_model": "./output/pth/TotalPulmonaryVessels/efficientnet-b4_Unet/clip-rotated-bce-adam-customLR1-32x-wd0.0001/0602_142825/epoch_0602_142825_21",
        # "best_model": "./output/pth/TotalPulmonaryVessels/stdc2_Unet/noclip-rotated-bce-adam-customLR1-32x-wd0/0605_172744/epoch_0605_172744_29",
        # "best_model": "./output/pth/TotalPulmonaryVessels/stdc2_Unet/noclip-rotated-bce-adam-customLR1-32x-wd0.0001/0606_170958/epoch_0606_170958_40",
        # "best_model": "./output/pth/TotalPulmonaryVessels/stdc2_Unet/noclip-rotated-bce-adam-customLR1-32x-wd0.0001/0606_182216/epoch_0606_182216_75",
        # "best_model": "./output/pth/TotalPulmonaryVessels/stdc2_Unet/noclip-rotated-bce-adam-customLR1-32x-wd0.0001/0607_110537/epoch_0607_110537_68",
        # "best_model": "./output/pth/TotalPulmonaryVessels/stdc2_Unet/noclip-rotated-bce-adam-customLR1-32x-wd0/0607_150441/epoch_0607_150441_58",
        # "best_model": "./output/pth/TotalPulmonaryVessels/stdc2_Unet/noclip-rotated-bce-adam-customLR1-32x-wd0.0005/0607_165658/epoch_0607_165658_55",
        # "best_model": "./output/pth/TotalPulmonaryVessels/stdc2_Unet/noclip-rotated-bce-adam-customLR1-32x-wd0.0001/0609_155826/epoch_0609_155826_49",

        ## Skin
        # "best_model": "./output/pth/Skin/Stdc2_Unet/clip-rotated-bce-adam-customLR1-32x-wd0.0001/0523_104343/epoch_0523_104343_6",
        # "best_model": "./output/pth/Skin/Stdc2_Unet/noclip-rotated-bce-adam-customLR1-32x-wd0/0606_174759/epoch_0606_174759_19",
        # "best_model": "./output/pth/Skin/Stdc2_Unet/noclip-rotated-bce-adam-customLR1-32x-wd0/0607_144240/epoch_0607_144240_40",
        # "best_model": "./output/pth/Skin/Stdc2_Unet/noclip-rotated-bce-adam-customLR1-32x-wd0/0608_093100/epoch_0608_093100_12",
        # "best_model": "./output/pth/Skin/Stdc2_Unet/noclip-rotated-bce-adam-customLR1-32x-wd0.0001/0608_133754/epoch_0608_133754_13",
        # "best_model": "./output/pth/Skin/Stdc2_Unet/noclip-rotated-bce-adam-customLR1-32x-wd0.0001/0608_160826/epoch_0608_160826_7",
        # "best_model": "./output/pth/Skin/stdc2_Unet/noclip-rotated-bce-adam-customLR1-32x-wd0/0609_174750/epoch_0609_174750_12",
        # "best_model": "./output/pth/Skin/stdc2_Unet/noclip-rotated-bce-adam-customLR1-32x-wd0.0001/0609_175618/epoch_0609_175618_3",


        # "stragety": "clip-rotated-dice-focal-adam", #"no-clip", #"clip-rotated"
        # "stragety": "clip-rotated-bce-adam-wd0",
        # "stragety": "clip-rotated-focal-adam",
        # "stragety": "clip-rotated-bce-adam-customLR1-32x-wd0",
        # "stragety": "clip-rotated-focal-adam-customLR1-32x-wd0",
        # "stragety": "clip-rotated-bce-adam-customLR1-32x-wd0.0001",
        # "stragety": "clip-rotated-bce-adam-customLR1-32x-wd0.0005",
        # "stragety": "clip-rotated-focal-adam-customLR1-32x-wd0.0001",
        # "stragety": "clip-rotated-16x-customLR1",
        "stragety": "clip-rotated-focal-adam-customLR1-16x-wd0.0001",
        # "stragety": "clip-rotated-focal-adam-customLR1-16x-wd0",
        # "stragety": "clip-rotated-bce-adam-customLR1-16x-wd0.0001",
        # "stragety": "noclip-rotated-bce-adam-customLR1-32x-wd0.0001",
        # "stragety": "noclip-rotated-bce-adam-customLR1-32x-wd0",
        # "stragety": "noclip-rotated-bce-adam-customLR1-32x-wd0.0005",
        # "stragety": "clip-rotated-tverskyfocal-adam-customLR1-16x-wd0.0001",

        "decoder_name": "Unet", ##"AttentionUnet", # "MANet", #
        "decoder_attention_type": None, #"scse"

        "device": "cuda:1",

        "test_uid_file": "/home/th/Data/data00/dst/500_100/valid.txt",
        "test_base_path": "/home/th/Data/data00/dst/",

        # "test_uid_file": "/home/th/Data/data03/dst/450_176/valid.txt",
        # "test_base_path": "/home/th/Data/data03/dst/",

        # "image_path": "/home/th/Data/dulicui/project/data/test/Bronchial/20190411-000053/dicom/",
        
        # "image_base_path": "/home/th/Data/dulicui/project/data/test/Bronchial/",
        # "image_base_path": "/home/th/Data/data_test/",  ### zhiqiguan
        # "image_base_path": "/home/th/Data/dulicui/project/data/test/Sphere/",

        "image_base_path": "/home/th/Data/data_sphere_test/",
        # "image_base_path": "/home/th/Data/data00/dst/dicom/",
        # "image_base_path": "/mnt/Data/data_sphere_test/",

        # "windowlevel": -850,
        # "windowwidth": 310, ## bronchial

        "windowlevel": -600,
        "windowwidth": 2000, ## Lung and sphere

        # "windowlevel": 75,
        # "windowwidth": 800, # PulmonaryVessels

        # "windowlevel": 75,
        # "windowwidth": 350, # Total

        # "windowlevel": 50,
        # "windowwidth": 500, # Heart

        # "windowlevel": 350,
        # "windowwidth": 800, ## Bone

        # "windowlevel": 135,
        # "windowwidth": 385, ## Skin

        "sigmoid_threshold": 0.4, #0.5,  0.7, 0.3, 0.4
        
        "output_stride": 16,
        "middle_patch_size": 512,
        "classes": ["Sphere"],  #["Skin"],#["Heart"], # ["TotalPulmonaryVessels"], #["Bone"],  # # #["PulmonaryVessels"], #["Bronchial"], #,["Lung"], #["lung_lu", "lung_ld", "lung_ru", "lung_rm",  "lung_rd"], #, #, #
        "in_channels": 1,
        "num_workers": 4,  # 多线程加载所需要的线程数目
        "pin_memory": True,  # 数据从CPU->pin_memory—>GPU加速
        "batch_size": 4,
        "save_base_path": "/home/th/Data/dulicui/project/git/segmentation_models.pytorch/output/stl",

        "encoder_depth": 5,
        "decoder_channels": [256, 128, 64, 32, 16],


        ## preprocess
        # "aug_name": "noclip-rotated",
        "aug_name": "clip-rotated",
    } 

    # import pdb; pdb.set_trace()
    # test_dataset = smp.datasets.SegDataset(
    #     uid_file=entrance["test_uid_file"],
    #     base_path=entrance["test_base_path"],
    #     stl_names=entrance["classes"],
    #     height=entrance["middle_patch_size"],
    #     width=entrance["middle_patch_size"],
    #     # channels=entrance["in_channels"],
    #     windowlevel=entrance["windowlevel"],
    #     windowwidth=entrance["windowwidth"],
    #     transform=None,
    #     # transform=transform,
    #     status=False
    # )
    # test(test_dataset, **entrance)
    generate_stl(**entrance)
    print("Happy End!")
