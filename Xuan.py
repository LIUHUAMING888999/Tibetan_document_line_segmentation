#旋转
# -*- coding: utf-8 -*-

import cv2
import os
from math import *
import numpy as np


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


# 旋转angle角度，缺失背景白色（255, 255, 255）填充
def rotate_bound_white_bg(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    pix_border = get_pix_background(image)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    # borderValue 缺失背景填充色彩，此处为白色，可自定义
    return cv2.warpAffine(image, M, (nW, nH),
                          borderValue=pix_border)  # return cv2.warpAffine(image, M, (nW, nH),borderValue=(0,255,255))
    # borderValue 缺省，默认是黑色（0, 0 , 0）
    # return cv2.warpAffine(image, M, (nW, nH))


# root = "/media/data_2/everyday/0725/hz/"
# list_hz = os.listdir(root)
# for img_name in list_hz:
#     img_path = root + img_name

# img = cv2.imread('001.18.png')
# img=cv2.resize(img,(2000,1024))
# imgRotation = rotate_bound_white_bg(img, 45)
# cv2.imshow("img", img)
# cv2.imshow("imgRotation", imgRotation)
# cv2.waitKey(0)