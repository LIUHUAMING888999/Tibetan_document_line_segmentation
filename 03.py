# 根据分块行投影求出各区域近似基线
import math

import cv2
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from Radon1 import xuan_z
from skimage import transform
import argparse
from Xuan import *
from Find1 import *
from Xian_xuan import *

import matplotlib.pyplot as plt


def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")

args = vars(ap.parse_args())

# 初始化
img = cv2.imread(args["image"])
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show('img', img1)
img2 = cv2.resize(img1, (2000, 1024))
cv_show('img', img2)
# 反向二值

bw2 = xuan_z(args["image"])  # 切片专用
cv_show('bw', bw2)
bw = bw2 / 255

bw1 = 1 - bw
bw3 = 1 - bw1
cv_show('iimgbw1', bw1)

width = bw3.shape[0]
height = bw3.shape[1]
print(height, width)
point_w = []
point_b = []
# 统计每行像素值
for row in range(width):
    a = 0
    b = 0
    for col in range(height):
        if row < 100 or row > 900:
            val = bw3[50][50]
        else:
            val = bw1[row][col]
        # print(val)
        ###################
        # if val !=255:
        #     print(val)
        ##############
        if val.all() == 0:
            a = a + 1
        else:
            b = b + 1
    point_w.append(a)
    point_b.append(b)
    # print("第", row, "/", (width ), "行，黑色像素有", a, "个，白色像素有", b, "个")
    # print(row,b)
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

y_b = savgol_filter(point_b, 5, 1, mode='nearest')
y_w = savgol_filter(point_w, 5, 1, mode='nearest')

# 可视化图线
# plt.subplot(211)
# plt.plot( y_b, 'b', label = 'savgol')
# plt.title('feng')
# plt.subplot(212)
# plt.plot( y_w, 'b', label = 'savgol')
# plt.title('gu')
# plt.show()

# 输出各峰谷，峰值所在行
x_w = point_w[0:4000]
peaks_w, _ = find_peaks(x_w, height=1200, distance=65)
x_b = point_b[0:4000]
peaks_b, _ = find_peaks(x_b, height=500, distance=65)
print(peaks_b)
print(peaks_w)

# 根据波峰波谷进行切片
img_crop = bw2.copy()

img_c = bw2.copy()
# c=int(height-(0.125)*height)
# d=int(height-(0.875)*height)
# print(c,d)
# img_crop1=img_c[205:215,d:c]
# cv_show('igh',img_crop1)
for i in range(8):

    A = peaks_w[i]
    B = peaks_w[i + 1]
    C = peaks_b[i]
    if abs(C - A) <= 10:
        A -= 20
    if B - A > 120:
        B -= 9
    # imag=img_crop[A:B,:]
    # cv_show('img',imag)
    # A = peaks_w[0]
    # B=(peaks_w[-1]-A)//8#平均值
    # imag=img_crop[(A+B*i):(A+B*(i+1)),:]
    # cv_show('img',imag)
    for hang in range(width):
        if hang == C:
            for h in range(10):
                c = int(height - (0.125) * height)  # 切到行尾
                d = int(height - (0.875) * height)  # 切到行头
                e = int((c - d) / 10)  # 分成十份后，一份的宽
                f = d + e * (h + 1)
                g = d + e * h
                # print(c)
                # print(d)
                # print(e)
                # print(f)
                # img_crop1 = img_c[hang - 12:hang + 7, g:f]  # 切割波峰附近区域
                img_crop1 = img_c[hang - 20:hang + 7, g:f]  # 切割波峰附近区域
                T_m = []  # 收集一张图片的【最大像素所在行：像素值】
                T_t_a = []  # 收集每个角度的峰值
                T_t_angle = []  # 收集每个峰值对应的角度，以便找出最大峰值所对应的角度
                Collect_xy_start = []#收集旋转后的基线起点坐标
                Collect_xy_end = []#收集旋转后的基点结尾坐标
                for angle1 in np.arange(-0.5, 0.55, 0.05):  # 从（-5，5）步长为0.05
                    imgRotation = rotate_bound_white_bg(img_crop1, angle1)  # 旋转
                    # print(imgRotation.shape)
                    T_m = Tong_ji(imgRotation)  # 统计区域内波峰波谷，返回其所在行
                    # print(T_m)
                    T_m['angle'] = angle1
                    T_t_a.append(T_m)
                    # print(T_t_a)
                    # T_t_angle.append(angle1)
                    # cv_show('imgRotation',imgRotation)
                lis = [i['value'] for i in T_t_a]  # 遍历列表中字典的‘value'组成一个新列表
                max_lis = max(lis)
                max_lis_index = lis.index(max_lis)  # 找到最大值对应的索引
                find_dict = list(T_t_a[max_lis_index].values())  # 根据索引找到所在字典的所有值（行,最大像素值，角度）
                # print(find_dict)
                find_hang = find_dict[0]  # 区域最大值所在行
                find_angle = find_dict[-1]  # 角度
                find_height = find_dict[-2]  # 区域拥有行数
                XX = int(hang - 2)
                g_f = (g + f) / 2

                # print(XX)
                # print(find_height)
                # print(g_f)
                # print(find_angle)
                Collect_xy_start = onepoint(XX, g, find_hang, g_f, find_angle)
                # print(Collect_xy_start)
                XXX1 = Collect_xy_start[0]
                XXX1=int(XXX1-2*math.tan(find_angle))
                YYY1 = Collect_xy_start[1]
                YYY1=round(YYY1-5*math.tan(find_angle))
                Collect_xy_end = onepoint(XX, f, find_hang, g_f, find_angle)
                # print(Collect_xy_end)
                XXX2 = Collect_xy_end[0]

                XXX2=int(XXX2-2*math.tan(find_angle))
                #
                YYY2 = Collect_xy_end[1]
                YYY2=round(YYY2-5*math.tan(find_angle))
                # print(XX)
                # print(find_hang)
                # print(find_height)
                # cv2.line(img_c, (g, XX), (f, XX), (0, 0, 255), 1)#各基线未旋转前
                cv2.line(img_c, (YYY1, XXX1), (YYY2, XXX2), (0, 0, 255), 1)
cv_show('opi', img_c)
