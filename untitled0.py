import numpy as np
import os
import cv2
import math
def get_pix_background(img):
    T = 15
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]
    for h in range(height):
        for w in range(width):
            b = img[h, w, 0]
            g = img[h, w, 1]
            r = img[h, w, 2]
            if abs(b - g )< T and abs(b - r )< T and abs(g - r)< T:
               return (int(b), int(g), int(r))
    return (int(img[1, 1, 0]), int(img[1, 1, 1]), int(img[1, 1, 2]))


def rotate(image, angle, center=None, scale=1.0):
    image=cv2.resize(image,(2000,1024))
    (w, h) = image.shape[0:2]
    pix_border = get_pix_background(image)
    if center is None:
        center = (w // 2, h // 2)
    wrapMat = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, wrapMat, (h, w),borderValue=pix_border)


# 使用矩形框
def Xuan_zhuan(image_path):
    # 读取图片，灰度化
    src = cv2.imread(image_path)
    # cv2.imshow("src", src)
    # cv2.waitKey()
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("gray", gray)
    # cv2.waitKey()
    # 图像取非
    grayNot = cv2.bitwise_not(gray)
    # cv2.imshow("grayNot", grayNot)
    # cv2.waitKey()
    # 二值化
    threImg = cv2.threshold(grayNot, 100, 255, cv2.THRESH_BINARY )[1]
    # cv2.imshow("threImg", threImg)
    # cv2.waitKey()
    # 获得有文本区域的点集,求点集的最小外接矩形框，并返回旋转角度
    coords = np.column_stack(np.where(threImg > 0))
    angle = cv2.minAreaRect(coords)[-1]
    angle = angle - 90
    if angle < -45:
        angle = -(angle + 90)
    else:
        angle = -angle

    # 仿射变换，将原图校正
    dst = rotate(src, angle)
    # cv2.imshow("dst", dst)
    # cv2.waitKey()
    print(angle)
    return dst


# if __name__ == "__main__":
#     Xuan_zhuan()
