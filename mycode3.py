import cv2
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from Radon1 import xuan_z
from skimage import transform
import argparse

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

#初始化
img= cv2.imread(args["image"])
# img1= cv2.imread(args["image"],0)
img1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv_show('img',img1)
img2 = cv2.resize(img1, (2000, 1024))
cv_show('img', img2)
# 反向二值

bw2=xuan_z(args["image"])#切片专用
cv_show('bw',bw2)
bw = bw2 / 255

bw1 = 1 - bw
bw3=1-bw1
cv_show('img', bw1)

width = bw3.shape[0]
height = bw3.shape[1]
print( height,width)
point_w=[]
point_b=[]
#统计每行像素值
for row in range(width):
    a=0
    b=0
    for col in range(height):
        if row<100 or row>900:
            val=bw3[50][50]
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
#可视化原始图像峰值，波谷值
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

y_b= savgol_filter(point_b, 5, 1, mode= 'nearest')
y_w = savgol_filter(point_w, 5, 1, mode= 'nearest')

# 可视化图线
plt.subplot(211)
plt.plot( y_b, 'b', label = 'savgol')
plt.title('feng')
plt.subplot(212)
plt.plot( y_w, 'b', label = 'savgol')
plt.title('gu')
plt.show()

#输出各峰谷，峰值所在行
x_w = point_w[0:4000]
peaks_w, _ = find_peaks(x_w, height=1200,distance=65)
x_b = point_b[0:4000]
peaks_b, _ = find_peaks(x_b, height=500,distance=65)
print(peaks_b)
print(peaks_w)

# #根据波峰波谷进行切片
# img_crop=bw2.copy()
# for i in range(8):
#     A=peaks_w[i]#波谷
#     B=peaks_w[i+1]#波峰
#     C=peaks_b[i]
#     if abs(C-A)<=10:
#         A-=20
#     if B-A>120:
#         B-=9
#     imag=img_crop[A:B,:]
#     cv_show('img',imag)
    # A = peaks_w[0]
    # B=(peaks_w[-1]-A)//8#平均值
    # imag=img_crop[(A+B*i):(A+B*(i+1)),:]
    # cv_show('img',imag)

img_crop=bw1.copy()
for i in range(8):
    A=peaks_w[i]
    B=peaks_w[i+1]
    C=peaks_b[i]
    if abs(C-A)<=10:
        A-=20
    if B-A>120:
        B-=9
    img_crop[(C-3):(C+60),:]=0
cv_show('img',img_crop)
