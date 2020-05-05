"""
图片正畸, 切边操作
"""
import os
import numpy as np
import cv2
from skimage.transform import ProjectiveTransform
import sys


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
    img = cv2.imread(os.path.join(dirpath, filename))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    _, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)

    # cv2.imshow("", thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # print(hierarchy)

    # marker_w = template["pages"][0]["Marker"]["width"] * 17 / 18
    # marker_h = template["pages"][0]["Marker"]["height"] * 17 / 18

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
        # print(x_max, x_min, y_max, y_min)
        # print(x_max - x_min, y_max - y_min)
        if x_max - x_min < (marker_w - 8) / 2450 * img.shape[1] or x_max - x_min > (marker_w + 8) / 2450 * img.shape[1]\
             or y_max - y_min < (marker_h - 8) / 3496 * img.shape[0] or y_max - y_min > (marker_h + 8) / 3496 * img.shape[0]:
            continue
        elif x_max > 250 / 2450 * img.shape[1] and x_min < 2200 / 2450 * img.shape[1]\
             and y_max > 250 / 3496 * img.shape[0] and y_min < 3250 / 3496 * img.shape[0]:
            # print(x_min / img.shape[1] * 2450, y_min / img.shape[0] * 3496)
            continue

        # print((x_max - x_min) * 2450 / img.shape[1], (y_max - y_min) * 3496 / img.shape[0])
        # print(x_max * 2450 / img.shape[1], x_min * 2450 / img.shape[1], y_max * 3496 / img.shape[0], y_min * 3496 / img.shape[0])

        fcontour = np.array([[y_min, x_min], [y_min, x_max], [y_max, x_max], [y_max, x_min]])
        # print(fcontour)
        fcontours.append(fcontour)

    print(len(fcontours))

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

    # img = img[137:2237, 110:1554]
    # cv2.imshow("", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return np.array([[y0, x0], [y1, x1], [y2, x2], [y3, x3]])


def transform(img, u, v):
    """
    图片旋转

    Parameters
    ----------
    img : array_like
    u :
    v :

    Returns
    -------
    """
    t = ProjectiveTransform()
    t.estimate(u, v)
    res = np.zeros((u[2][0], u[2][1], 3))
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            pos = t(np.array([[i, j]]))[0]

            y = min(max(int(pos[0]), 0), img.shape[0] - 1)
            x = min(max(int(pos[1]), 0), img.shape[1] - 1)
            res[i][j] = img[y][x]
    return res


def process(raw_file_path, raw_file_name, save_dir):
    """
    旋转图片, 进行图片裁切, 然后压缩

    Parameters
    ----------
    raw_file_path : 原始文件路径
    save_dir : 裁切后的文件保存路径

    Returns
    -------
    new_file_path : str 处理后的文件路径
    """
    if (not(os.path.exists(save_dir))):
        os.mkdir(save_dir)

    img = cv2.imread(raw_file_path)

    new_file_name = raw_file_name + ".processed.jpg"
    new_file_path = os.path.join(save_dir, new_file_name)

    # TODO: what if marker data changes?
    #    Marker":{"width":90,"height":32,"x":2050,"y": 3162},
    # x = template["pages"][0]["Marker"]["x"] + template["pages"][0]["Marker"]["width"]
    # y = template["pages"][0]["Marker"]["y"] + template["pages"][0]["Marker"]["height"]

    x = 2140
    y = 3194

    v = findCorner(img)

    u = np.array([[0, 0], [0, x], [y, x], [y, 0]])
    print(u, v)
    img = transform(img, u, v)

    # 压缩图片
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    cv2.imencode('.jpg', img, encode_param)

    # 写入图片
    cv2.imwrite(new_file_path, img)

    return new_file_path

for (dirpath, dirnames, filenames) in os.walk("test_img 2"):
    for i in range(len(filenames)):
        filename = filenames[i]
        # print(filename)
        if filename.split(".")[-1] != "jpg":
            continue
        img = cv2.imread(os.path.join(dirpath, filename))

        process(os.path.join(dirpath, filename), filename, dirpath + "_transform")
        # if (filename == "BRN3C2AF43471C5_20191220_165928_075518.jpg"):
          # process(os.path.join(dirpath, filename), filename, dirpath + "_transform")
        # findCorner(img)


        
