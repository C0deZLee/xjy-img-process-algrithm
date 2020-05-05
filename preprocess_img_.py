"""
图片正畸, 切边操作
"""
import os
import numpy as np
import cv2
from PIL import Image

import sys

from imutils.perspective import four_point_transform
from imutils import contours


def findCorner(img):
    """
    找到图片的四个角

    Parameters
    ----------
    img : array_like

    Returns
    -------
    array : np.array[][]
        包含四个点的二维数组
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    _, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    marker_w = 85
    marker_h = 27

    y0 = img.shape[0]
    x0 = img.shape[1]
    y1 = img.shape[0]
    x1 = 0
    y2 = 0
    x2 = 0
    y3 = 0
    x3 = img.shape[1]

    fcontours = []

    for contour in contours:
        x_max = 0
        y_max = 0
        x_min = thresh.shape[1]
        y_min = thresh.shape[0]
        for point in contour:
            x_max = max(x_max, point[0][0])
            x_min = min(x_min, point[0][0])
            y_max = max(y_max, point[0][1])
            y_min = min(y_min, point[0][1])

        # print((x_max - x_min) * 2450 / img.shape[1], (y_max - y_min) * 3496 / img.shape[0])
        # print(x_max * 2450 / img.shape[1], x_min * 2450 / img.shape[1], y_max * 3496 / img.shape[0], y_min * 3496 / img.shape[0])

        if x_max - x_min < (marker_w - 8) / 2450 * img.shape[1] or x_max - x_min > (marker_w + 8) / 2450 * img.shape[1]\
             or y_max - y_min < (marker_h - 8) / 3496 * img.shape[0] or y_max - y_min > (marker_h + 8) / 3496 * img.shape[0]:
            continue
        elif x_max > 250 / 2450 * img.shape[1] and x_min < 2200 / 2450 * img.shape[1]\
             and y_max > 250 / 3496 * img.shape[0] and y_min < 3250 / 3496 * img.shape[0]:
            continue

        fcontour = np.array([[y_min, x_min], [y_min, x_max], [y_max, x_max], [y_max, x_min]])

        fcontours.append(fcontour)

    if (len(fcontours) != 4 and len(fcontours) != 3):
        print(len(fcontours))
        print("Algorithm failed")
        sys.exit()

    for fcontour in fcontours:
        if fcontour[0][0] < img.shape[0] / 2 and fcontour[0][1] < img.shape[1] / 2:
            y0 = fcontour[0][0]
            x0 = fcontour[0][1]
        elif fcontour[1][0] < img.shape[0] / 2 and fcontour[1][1] > img.shape[1] / 2:
            y1 = fcontour[1][0]
            x1 = fcontour[1][1]
        elif fcontour[2][0] > img.shape[0] / 2 and fcontour[2][1] > img.shape[1] / 2:
            y2 = fcontour[2][0]
            x2 = fcontour[2][1]
        elif fcontour[3][0] > img.shape[0] / 2 and fcontour[3][1] < img.shape[1] / 2:
            y3 = fcontour[3][0]
            x3 = fcontour[3][1]

    if (len(fcontours) == 3):
        if (x2 == 0 and y2 == 0):
            x2 = x1
            y2 = y3
        elif (x0 == img.shape[1] and y0 == img.shape[0]):
            x0 = x3
            y0 = y1
        elif (x1 == 0 and y1 == img.shape[0]):
            x1 = x2
            y1 = y0
        elif (x3 == img.shape[1] and y3 == 0):
            x3 = x0
            y3 = y2
        else:
            print(len(fcontours))
            print("Algorithm failed")
            sys.exit()

    return np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])

def filp_img(raw_file_path):
    """
    将给定图片旋转180度

    Parameters
    ----------
    raw_file_path : 原始文件路径

    Returns
    -------
    None
    """
    img = Image.open(raw_file_path) 
    img = img.rotate(180, Image.NEAREST, expand = 1) 
    img.save(raw_file_path)

def identify_page(img):
    """
    判断图片是第几页以及是否颠倒

    Parameters
    ----------
    img : open cv 图片实例

    Returns
    -------
    reverted : bool 是正的还是颠倒的, True是颠倒的, False是正的
    page : str 是A面还是B面, C面, D面 , 返回值为 [1, 2, 3, 4] 中的一个
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    _, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)

    # cv2.imshow("", thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    marker_w = 30
    marker_h = 30

    fcontours_top = []
    fcontours_bottom = []
    
    for contour in contours:
        x_max = 0
        y_max = 0
        x_min = thresh.shape[1]
        y_min = thresh.shape[0]
        for point in contour:
            x_max = max(x_max, point[0][0])
            x_min = min(x_min, point[0][0])
            y_max = max(y_max, point[0][1])
            y_min = min(y_min, point[0][1])
        if x_max - x_min >= marker_w - 4 and x_max - x_min <= marker_w + 15\
             and y_max - y_min >= marker_h - 10 and y_max - y_min <= marker_h + 15:
            if y_min > 3150:
                print(x_max - x_min, y_max - y_min)
                fcontour = np.array([[y_min, x_min], [y_min, x_max], [y_max, x_max], [y_max, x_min]])
                fcontours_bottom.append(fcontour)
            elif y_max < 44:
                print(x_max - x_min, y_max - y_min)
                fcontour = np.array([[y_min, x_min], [y_min, x_max], [y_max, x_max], [y_max, x_min]])
                fcontours_top.append(fcontour)

    # print(fcontours_bottom, fcontours_top)
    if len(fcontours_bottom) == 0:
        return False, len(fcontours_top)

    return True, len(fcontours_bottom)

def process(raw_file_path, raw_file_name):
    """
    旋转图片, 进行图片裁切, 然后压缩

    Parameters
    ----------
    raw_file_path : 原始文件路径
    raw_file_name : 原始文件名称

    Returns
    -------
    reverted : bool 是正的还是颠倒的, True是颠倒的, False是正的
    page : str 是A面还是B面, C面, D面 , 返回值为 [1, 2, 3, 4] 中的一个
    """
    img = cv2.imread(raw_file_path)

    # TODO: what if marker data changes?
    #    Marker":{"width":90,"height":32,"x":2050,"y": 3162},
    # x = template["pages"][0]["Marker"]["x"] + template["pages"][0]["Marker"]["width"]
    # y = template["pages"][0]["Marker"]["y"] + template["pages"][0]["Marker"]["height"]

    x = 2140
    y = 3194

    # 寻找图片的四个边界
    v = findCorner(img)

    # 图片正畸
    img = four_point_transform(img, v)

    img = cv2.resize(img, (x, y))

    # print(img.shape)

    # 压缩图片
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    cv2.imencode('.jpg', img, encode_param)

    # 覆盖原始图片
    # cv2.imwrite(raw_file_path, img)

    # 确认图片页数和颠倒情况
    reverted, page = identify_page(img)
    print(reverted, page)

    # 如果颠倒, 则翻转图片并覆盖
    if not(reverted):
        filp_img(raw_file_path)

    return reverted, page

# img = cv2.imread("E77673E9X112769_20191119_074637_0125.min.jpg")
# identify_page(img)


# for (dirpath, dirnames, filenames) in os.walk("write_transform"):
#     for i in range(len(filenames)):
#         filename = filenames[i]
#         # print(filename)
#         if filename.split(".")[-1] != "jpg":
#             continue
#         img = cv2.imread(os.path.join(dirpath, filename))
#         print(identify_page(img))

        # process(os.path.join(dirpath, filename), filename)

