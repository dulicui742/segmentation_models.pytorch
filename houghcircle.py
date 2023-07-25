import vtk
from vtk.util import numpy_support
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pyclipper



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
    plt.show()
    # plt.pause(1)


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
    # import pdb; pdb.set_trace()

    # solution = []
    # for i in range(len(solution)):
    #     solution.append(np.array(solution[i]).reshape(-1, 1, 2).astype(int))


    ### dulicui 
    # if len(solution) != 1:
    #     solution = np.array(solution[0]).reshape(-1, 1, 2).astype(int)
    # else:
    #     solution = np.array(solution).reshape(-1, 1, 2).astype(int)


    solution = max(solution, key=lambda k: len(k)) ## 获取长度最大的solution
    solution = np.array(solution).reshape(-1, 1, 2).astype(int)

    return solution

# image_path = "/mnt/Data/data_sphere_test/39/dicom/"
image_path = "D:\\project\\TrueHealth\\20230217_Alg1\\data\\examples\\sphere_test\\dicom_39"
# image_path = "D:\\project\\TrueHealth\\20230217_Alg1\\data\\examples\\sphere_test\\dicom_53"
# # image_path = "D:\\project\\TrueHealth\\20230217_Alg1\\data\\examples\\sphere_test\\dicom_22"
# image_path = "D:\\project\\TrueHealth\\20230217_Alg1\\data\\examples\\sphere_test\\dicom_47"
# image_path = "D:\\project\\TrueHealth\\20230217_Alg1\\data\\examples\\sphere_test\\dicom_18"

windowlevel = -600
windowwidth = 2000
# windowlevel = 0
# windowwidth = 2048


dicomreader = vtk.vtkDICOMImageReader()
dicomreader.SetDirectoryName(image_path)
dicomreader.Update()
output = dicomreader.GetOutput()
dimensions = output.GetDimensions()
print("dimension:", dimensions)

dicomArray = numpy_support.vtk_to_numpy(output.GetPointData().GetScalars())
dicomArray = dicomArray.reshape(dimensions[::-1]).astype(np.float32)
# dicomArray = dicomArray.reshape(dimensions[2], dimensions[0], dimensions[1]).astype(np.float32)
copyArray = dicomArray * 0

# import pdb; pdb.set_trace()
for i in range(dimensions[2]):
    # if i < 78: ## for dicom_39
    if i < 130:
    # if i < 9: ##for dicom_53
    # if i < 97:
    # # if i < 115:
    # # if i < 102: ### for 22
        continue
    print("slice:", i)
    img = dicomArray[i, ::-1, :]  # slice
    # img = ((img - windowlevel) / windowwidth + 0.5) * 255

    ## for test
    img = ((img - windowlevel) / windowwidth + 0.5) 
    img = np.clip(img, 0, 1) * 255
    img_bk = img.copy()

    ### ===================contour=================
    # import pdb; pdb.set_trace()
    #img_bk = img.copy()
    img = img.astype(np.uint8)
    # gaussian = cv2.GaussianBlur(img, (7, 7),0)
    # edges = cv2.Canny(gaussian, 100, 200)
    gaussian = cv2.medianBlur(img, 3)
    # ret, edges =  cv2.threshold(gaussian, 127, 255, cv2.THRESH_BINARY)
    ret, edges =  cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    edges_ = edges.copy()

    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    edges = opening

    # #设置卷积核
    # kernel = np.ones((7,7), np.uint8)
    
    # #图像开运算
    # result = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #cv2.RETR_TREE

    
    # cv2.drawContours(imgrgb, contours, -1, (0, 255, 0), thickness=1)
    
    # import pdb; pdb.set_trace()
    # idx = 0
    # area = 0
    # for i in range(len(contours)):
    #     ## area
    #     # area = cv2.contourArea(contours[i])
    #     # if area > 400:
    #     #     continue

    #     ## rect
    #     # rect = cv2.minAreaRect(contours[i])
    #     # w, h =  rect[1]
    #     # if w == 0 or h == 0:
    #     #     continue
    #     # if w/h > 2 or h/w > 2:
    #     #     # print(w, h)
    #     #     continue
        

    #     ## circle
    #     cur_area = cv2.contourArea(contours[i])
    #     if cur_area > area:
    #         idx = i
    #         area =  cur_area
    
    area = []
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    max_idx = np.argmax(np.array(area))
    idx = max_idx
        
    (x,y), radius = cv2.minEnclosingCircle(contours[idx])
    
    center = (int(x), int(y))
    radius = int(radius)
    print(f"center: {center}, radius: {radius}")

    # import pdb; pdb.set_trace()
    edges[edges > 0] = 1
    img = img * edges

    # import pdb; pdb.set_trace()
    new = np.zeros(img.shape, dtype=np.uint8)
    newrgb = cv2.cvtColor(new, cv2.COLOR_GRAY2BGR)

    outmask1 = np.zeros(img.shape, dtype=np.uint8)
    outmaskrgb1 = cv2.cvtColor(outmask1, cv2.COLOR_GRAY2BGR)

    outmask2 = np.zeros(img.shape, dtype=np.uint8)
    outmaskrgb2 = cv2.cvtColor(outmask2, cv2.COLOR_GRAY2BGR)

    imgrgb =  cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # cv2.circle(imgrgb, center, radius, (0, 255, 0), 2)
    # cv2.circle(newrgb, center, radius, (0, 255, 0), -1)
    # cv2.circle(imgrgb, center, radius+20, (0, 0, 255), 2) ###向外扩10+pixel

    # if i == 82:
    #     import pdb; pdb.set_trace()

    # if i == 249:
    #     import pdb; pdb.set_trace()
    contour_extend = equidistant_zoom_contour(contours[idx], 40)
    contour_inside = equidistant_zoom_contour(contours[idx], -10)
    cv2.drawContours(imgrgb, contours[idx], -1, (255, 0, 0), thickness=2)
    cv2.drawContours(imgrgb, [contour_extend], -1, (255, 255, 0), thickness=2)
    cv2.drawContours(imgrgb, [contour_inside], -1, (0, 255, 255), thickness=2)
    # cv2.drawContours(imgrgb, contour_extend, -1, (255, 255, 0), thickness=2)
    # cv2.drawContours(imgrgb, contour_inside, -1, (0, 255, 255), thickness=2)

    # import pdb; pdb.set_trace()
    # cv2.drawContours(outmaskrgb1, contours[idx], -1, (255, 0, 0), -1)
    # mask1 = cv2.drawContours(outmaskrgb1, contours, idx, 1, cv2.FILLED)

    ##---------------
    mask1 = cv2.drawContours(outmaskrgb1, [contour_extend], -1, 1, cv2.FILLED)
    mask2 = cv2.drawContours(outmaskrgb2, [contour_inside], -1, 1, cv2.FILLED)

    # mask1 = cv2.drawContours(outmaskrgb1, contour_extend, -1, 1, cv2.FILLED)
    # mask2 = cv2.drawContours(outmaskrgb2, contour_inside, -1, 1, cv2.FILLED)
    ##---------------

    # mask2 = cv2.drawContours(outmaskrgb2, [contours[idx]], -1, 1, cv2.FILLED)
    # mask1 = cv2.drawContours(imgrgb, contours, idx, 1, cv2.FILLED)

    # rect = cv2.minAreaRect(contours[idx])
    # points = cv2.boxPoints(rect)  # 得到最小外接矩形的四个点坐标
    # points = np.int0(points)  # 坐标值取整
    # cv2.drawContours(imgrgb, [points], 0, (0, 0, 255), 2)
    # cv2.drawContours(newrgb, [points], 0, (0, 0, 255), 2)

    # newrgb[newrgb > 0] = 1
    # mask = newrgb[:,:,0]
    mask = cv2.cvtColor(newrgb, cv2.COLOR_BGR2GRAY)
    mask[mask > 0] = 1

    # mask1 = cv2.cvtColor(outmaskrgb1, cv2.COLOR_BGR2GRAY)
    # mask1[mask1 > 0] = 1
    # mask1 = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
    mask1 = mask1[:,:,0]
    mask2 = mask2[:,:,0]
    cmask = mask1 - mask2

    # img = img * mask
    # img1 = img * mask1
    # if i == 1:
    #     import pdb; pdb.set_trace()

    img1 = img_bk * cmask * edges
    # img1 = img_bk * mask1
    img2 = img1.copy()
    # img2[1-cmask] = 100
    img2[img2 == 0] = 80

    # img2[img2 !=127] = 255
    print(img2.shape)

    # import pdb; pdb.set_trace()
        # cv2.drawContours(imgrgb, [contours[i]], -1, (0, 255, 0), thickness=1)
    # params = {"gaussian": gaussian, "binary": edges, "image": imgrgb, "ori": img, "mask": mask, "i1": img1, "mask1": mask1}
    # if i == 28:
    # params = {"ori-image": img_bk, "binary": edges_, "op": opening, "contour": imgrgb,  "mask": cmask, "fg": img1,"fg1": img2, "m1": mask1, "m2": mask2}
    params = {"ori-image": img_bk, "binary": opening, "contour": imgrgb,  "mask": cmask*edges, "fground": img1, }
    visualize(**params)
    ### ===================contour=================


    # ### ===================houghcircle=================
    # # ### 
    # img = img.astype(np.uint8)
    # gaussian = cv2.GaussianBlur(img, (7, 7), 0)
    # # edges = cv2.Canny(gaussian, 100, 200)
    # # gaussian = cv2.medianBlur(img, 3)
    # # ret, edges =  cv2.threshold(gaussian, 127, 255, cv2.THRESH_BINARY)
    # # gaussian = cv2.medianBlur(img, 7)

    # # params = {"image": img}
    # # visualize(**params)

    # # circles = cv2.HoughCircles(image, method, dp, minDist[, param1[, param2[, minRadius[, maxRadius]]]]])
    
    # # Below are the parameters explained in detail

    # # image: 8-bit, single-channel, grayscale input image
    # # method: HOUGH_GRADIENT and HOUGH_GRADIENT_ALT
    # # dp: The inverse ratio of accumulator resolution and image resolution
    # # minDist: Minimum distance between the centers of the detected circles. All the candidates below this distance are neglected as explained above
    # # param1: it is the higher threshold of the two passed to the Canny edge detector (the lower canny threshold is twice smaller)
    # # param2: it is the accumulator threshold for the circle centers at the detection stage as discussed above.
    # # minRadius: minimum radius that you expect. If unknown, put zero as default.
    # # maxRadius: if -ve, only circle centers are returned without radius search. If unknown, put zero as default.
    
    # circles = cv2.HoughCircles(
    #     gaussian, 
    #     # edges,
    #     cv2.HOUGH_GRADIENT, 
    #     1, 
    #     30, 
    #     param1=50, 
    #     param2=13, 
    #     minRadius=0,
    #     maxRadius=15
    #     )
    # # params = {"gaussian": gaussian, "canny": edges, "image": img, }
    # # visualize(**params)
    # # import pdb; pdb.set_trace()
    # if circles is None:
    #     continue
    # circles = np.uint16(np.around(circles))
    # for i in circles[0,:]:
    #     cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # # cv2.imshow("img", img/255)
    # params = {"ori_image": img_bk, "gaussian": gaussian, "image": img, }
    # visualize(**params)
    # ### ===================houghcircle=================


def foreground(img):
    img = img.astype(np.uint8)
    gaussian = cv2.medianBlur(img, 3)
    # ret, edges = cv2.threshold(gaussian, 127, 255, cv2.THRESH_BINARY)
    ret, edges =  cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(
        edges, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    ) #cv2.RETR_TREE

    idx = 0
    area = 0
    for i in range(len(contours)):
        ## get contour idx with max area
        cur_area = cv2.contourArea(contours[i])
        if cur_area > area:
            idx = i
            area = cur_area
        
    (x, y), radius = cv2.minEnclosingCircle(contours[idx])
    center = (int(x), int(y))
    radius = int(radius)

    new = np.zeros(img.shape, dtype=np.uint8)
    imgrgb =  cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    newrgb = cv2.cvtColor(new, cv2.COLOR_GRAY2BGR)
    cv2.circle(imgrgb, center, radius, (0, 255, 0), 1)
    cv2.circle(newrgb, center, radius, (0, 255, 0), -1)

    mask = cv2.cvtColor(newrgb, cv2.COLOR_BGR2GRAY)
    mask[mask > 0] = 1
    edges[edges > 0] = 1
    mask *= edges ## 二值化和外接圆共同确定前景
    img = img * mask
    return img



