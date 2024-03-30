'''import cv2

# 读取图片并转至灰度模式
imagepath = '001.18.png'
# imagepath=cv2.resize(imagepath,(2000,1024))
img = cv2.imread(imagepath, 1)
img=cv2.resize(img,(2000,1024))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 二值化
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# 图片轮廓
contours, hierarchy = cv2.findContours(thresh, 2, 1)
cnt = contours[0]
# 寻找凸包并绘制凸包（轮廓）
hull = cv2.convexHull(cnt)
print(hull)

length = len(hull)
for i in range(len(hull)):
    cv2.line(img, tuple(hull[i][0]), tuple(hull[(i+1)%length][0]), (0,255,0), 3)

# 显示图片
cv2.imshow('line', img)
cv2.waitKey()
'''
'''
打算利用轮廓检测进行大致行提取，先用大轮廓计算每行大概位置
import cv2
import matplotlib.pyplot as plt
def cv_show(name,image):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
img = cv2.imread("001.18.png",0)
img=cv2.resize(img,(2000,1024))
cv_show('img',img)
#反向二值
bw=img/255
bw1=1-bw
cv_show('img',bw1)

#统计每行像素值
hs=sum(bw1)
plt.hist(hs)
hight=hs.shape[0]
print(hight)
plt.show()
#求峰值
dis = 25
hs2= hs*0
i = dis + 1
y=hight - dis
for i in range(y) :
   hs2= sum(hs[i-dis:i+dis,:].shape[0])/( dis*2 + 1 )
   plt.hist(hs2)
   plt.show()'''

# 导入工具包
from imutils import contours
from scipy import ndimage
import numpy as np
import argparse
import cv2
import myutils  # 直接alt+shift+enter新建
from scipy.spatial import ConvexHull, convex_hull_plot_2d
# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")

args = vars(ap.parse_args())


def DiscreteRadonTransform(image, steps):
    channels = len(image[0])
    res = np.zeros((channels, channels), dtype='float64')
    for s in range(steps):
        rotation = ndimage.rotate(image, -s * 180 / steps, reshape=False).astype('float64')
        # print(sum(rotation).shape)
        res[:, s] = sum(rotation)
    return res

from Radon1 import xuan_z
# 绘图展示
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(10)
    cv2.destroyAllWindows()


# 读取输入图像，预处理
image = cv2.imread(args["image"])
cv_show('image', image)
imag = cv2.resize(image, (2000, 1024))
img = cv2.imread(args["image"], 0)



imgg = cv2.resize(img, (2000, 1024))
cv_show('img', imgg)
# 反向二值
bw = imgg / 255
gray = 1 - bw

cv_show('img', gray)
height = imgg.shape[0]
width = imgg.shape[1]
print(f'{width}{height}')

# 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3))  # 9,3%11,3比较好%13,3
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
k = np.ones((2, 3), np.uint8)
k1 = np.ones((3, 3), np.uint8)
dst1 = cv2.erode(gray, k, iterations=3)
dst = cv2.dilate(dst1, k1, iterations=3)

cv_show('img1', dst)
# open = cv2.morphologyEx(dst, cv2.MORPH_OPEN, k)
# cv_show('img2', open)
# 礼帽操作，突出更明亮的区域
tophat = cv2.morphologyEx(dst, cv2.MORPH_TOPHAT, rectKernel, iterations=5)
cv_show('tophat', tophat)

# 通过闭操作（先膨胀，再腐蚀）将数字连在一起
thresh = cv2.morphologyEx(tophat, cv2.MORPH_CLOSE, rectKernel, iterations=9)
cv_show('gradX', thresh)

# 计算轮廓

threshCnts, hierarchy = cv2.findContours(np.uint8(thresh.copy()), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

cnts =threshCnts

# print('iji',cnts)
print(len(cnts[1]))
print(cnts[1][1][0][0])
# print(hierarchy)

cur_img = imag.copy()
for h in range(len(cnts)):#轮廓个数
    for j in range(len(cnts[h])):#每个轮廓拥有的坐标
         if cnts[h][j][0][1]<480 and cnts[h][j][0][1]>455:
              cv2.line(cur_img,(cnts[h][j][0][0],cnts[h][j][0][1]),(cnts[h][j][0][0],cnts[h][j][0][1]), (0, 0, 255), 3)#(X,Y)(列，行）
cv2.imshow('img', cur_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
locs = []

'''
'''
# # 遍历轮廓
# for (i, c) in enumerate(cnts):
#     # 计算矩形
#     (x, y, w, h) = cv2.boundingRect(c)
#     ar = w / float(h)
#
#     # 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
#     if ar > 1 and ar < 40000:
#
#         if (w > 800 and w < 2000) and (h > 15 and h < 4000):
#                 # 符合的留下来
#                 locs.append((x, y, w, h))
#
# # 将符合的轮廓从左到右排序
# locs = sorted(locs, key=lambda x: x[0])
# output = []
#
# # 遍历每一个轮廓中的数字
# for (i, (gX, gY, gW, gH)) in enumerate(locs):
#     # initialize the list of group digits
#     groupOutput = []
#
#     # 根据坐标提取每一个组
#     group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
#     cv_show('group', group)
#
#     # 画出来
#
#     cv2.rectangle(imag, (gX - 5, gY - 5),
#                   (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
#     cv2.putText(imag, "".join(groupOutput), (gX, gY - 15),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
#
#     # 得到结果
#     output.extend(groupOutput)
#
# # 打印结果
#
# print("Credit Card #: {}".format("".join(output)))
# cv2.imshow("Image", imag)
# cv2.waitKey(0)

'''
'''
