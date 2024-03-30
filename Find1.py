'''
给个图片输出像素最大值所在行（波峰，波谷）
'''
import cv2
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from Radon1 import xuan_z
from skimage import transform
import argparse
from Xuan import *

import matplotlib.pyplot as plt
def Tong_ji(image):

    # ret, image = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
    width = image.shape[0]
    height = image.shape[1]
    # print( height,width)
    point_w=[]
    point_b=[]
    point_row=[]
    point_a=[]
    #统计每行像素值
    for row in range(width):
        a=0
        b=0
        for col in range(height):
            # if row<100 or row>900:
            #     val=image[50][50]
            # else:
            #     val = image[row][col]
            val = image[row][col]
            # print(val)
            ###################
            # if val !=255:
            #     print(val)
            ##############
            if val.all() == 0:
                a = a + 1
            else:
                b = b + 1
        point_a.append(a)
        point_row.append(row)
    max_a=max((point_a),default=0)#图片中最大的行黑像素点
    max_a_xia=point_a.index(max(point_a))#图片中最大的行黑像素点所对应的索引
    max_a_row=point_row[max_a_xia]#根据图片中最大的行黑像素点所对应的索引，求行的值
    # print(max_a)
    # print(max_a_xia)
    # # print(max_a_row)
    # print({max_a_row:max_a})
    return {'row':max_a_row,'value':max_a,'width':width}#{行，像素值}

        #
        # point_w.append(b)
        # point_b.append(a)

    #     print("第", row, "/", (width ), "行，黑色像素有", a, "个，白色像素有", b, "个")
    #     print(row,b)
    # 可视化原始图像峰值，波谷值
    # plt.subplot(211)
    # plt.plot(point_b)
    # plt.subplot(212)
    # plt.plot(point_w)
    # plt.xlabel('a')
    # plt.ylabel('b')
    # plt.title('as')
    # plt.tight_layout()
    #
    # plt.show()
    # plt.savefig('./catvsdog_AlexNet.jpg', dpi=200)

    # 使用Savitzky-Golay 滤波器后得到平滑图线

    # y_b= savgol_filter(point_b, 5, 1, mode= 'nearest')
    # y_w = savgol_filter(point_w, 5, 1, mode= 'nearest')

    # # 可视化图线
    # plt.subplot(211)
    # plt.plot( y_b, 'b', label = 'savgol')
    # plt.title('feng')
    # plt.subplot(212)
    # plt.plot( y_w, 'b', label = 'savgol')
    # plt.title('gu')
    # plt.show()

    #输出各峰谷，峰值所在行
    # x_w = point_w[0:4000]
    # peaks_w, peaks_value_w = find_peaks(x_w, height=0,distance=65)
    # x_b = point_b[0:4000]
    # peaks_b, peaks_value_b = find_peaks(x_b, height=0,distance=65)
    # a=peaks_value_b.values()
    # b=a.index(max(list(a)))
    # print(b)
    # c=peaks_b[b]
    # return c,int(max(list(a)))




# img=cv2.imread('006.365.png')
# mn=[]
# mn=Tong_ji(img)
# print(mn)

def Tong_ji_gu(image):

    # ret, image = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
    width = image.shape[0]
    height = image.shape[1]
    # print( height,width)
    point_w=[]
    point_b=[]
    point_row=[]
    point_a=[]
    #统计每行像素值
    for row in range(width):
        a=0
        b=0
        for col in range(height):
            # if row<100 or row>900:
            #     val=image[50][50]
            # else:
            #     val = image[row][col]
            val = image[row][col]
            # print(val)
            ###################
            # if val !=255:
            #     print(val)
            ##############
            if val.all() == 0:
                a = a + 1
            else:
                b = b + 1
        point_a.append(b)
        point_row.append(row)
    max_a=max((point_a),default=0)#图片中最大的行白像素点
    max_a_xia=point_a.index(max(point_a))#图片中最大的行白像素点所对应的索引
    max_a_row=point_row[max_a_xia]#根据图片中最大的行白像素点所对应的索引，求行的值
    # print(max_a)
    # print(max_a_xia)
    # # print(max_a_row)
    # print({max_a_row:max_a})
    return {'row':max_a_row,'value':max_a,'width':width}#{行，像素值}

