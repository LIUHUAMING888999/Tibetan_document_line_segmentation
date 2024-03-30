# function[baseLineImage] = find_syllable_line(I)
# # SYLLABLE_LINE
# 此处显示有关此函数的摘要
# # 此处显示详细说明
# # I: 文字的二值化图像，0
# 表示背景，1
# 表示前景（文字区域）
# imgsize = size(I)

# 使用水平直线检测器进行直线检测　只保留下黑　上白的直线
import cv2
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from skimage import transform
import argparse
import matplotlib.pyplot as plt
import scipy
from skimage import measure, color
from Gendist import gendist

def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
I=plt.imread('001.18.png',0)

# labels = measure.label(I, connectivity=1)
# print(labels)
# dst = color.label2rgb(labels)
# dst=cv2.resize(dst,(2000,1024))
# cv_show('dst',dst)

I=cv2.resize(I,(300,300))
I=I/255
h = np.zeros((3,20))
print(h)

h[:1,:] = -1
h[2:3,:] = 1
print(h)
Y = scipy.signal.correlate2d(h, I)#二维数字滤波器
[y,_]=Y.shape#[y, ~] = size(Y)

Y[y - 2: y,:] = 0
print(Y)
ymax =Y.flat[abs(Y).argmax()]
# Y=float(Y)
y = Y/ ymax
y1 = abs(y* (y < -0.5))

if y1.all()>0:
    y1=1
# print(y1)
'''# 进行连通区域分析
cc = bwconncomp(y1)
graindata = regionprops(cc, {'Extrema', 'Centroid', 'Area', 'BoundingBox', 'PixelIdxList'})
maxvalue = 99999'''
labels = measure.label(y1, connectivity=1)
print(labels)
# dst = color.label2rgb(labels)
dst=cv2.resize(I,(2000,1024))
cv_show('dst',dst)

graindata = measure.regionprops(labels,{'Extrema', 'Centroid', 'Area', 'BoundingBox', 'PixelIdxList'})
print('asd',graindata)
maxvalue = 99999

# 得到距离矩阵
dm = gendist(graindata, maxvalue)
print(dm)
'''
# 进行追踪
tc = GetTrackMatrix(dm, maxvalue, graindata, I)

# 根据追踪结果得到基线
[clp, cl, lineNum, dify, baseLineImage] = GetBaseLine(tc, graindata, y1)
# RGB = label2rgb(baseLineImage)
# figure, imshow(RGB)


function[clp, cl, lineNum, dify, baseLineImage] = GetBaseLine(tc, graindata, oriimg)
# 根据追踪结果得到从上到下排序好的基线
# 返回值
clp
一个cell数组
保存了每条基线的坐标点位置
# cl
按照x
y坐标的形式保存了每条基线的位置
# lineNum
基线数目
# avdify
保存了每条基线之间的距离
# baseLineImage
基线图

# 对追踪结果根据y坐标值进行排序
[lineNum, w] = size(tc)
ct = [graindata.Centroid]
oriy = zeros(1, lineNum)
for i=1:lineNum
idx = tc(i, 1)
oriy(i) = ct(2 * idx)

[~, idx] = sort(oriy, 'asc')
tc = tc(idx,:)

ii = zeros(size(oriimg))
for i = 1:lineNum
for j = 1:w
if tc(i, j) == 0
    break

ii(graindata(tc(i, j)).PixelIdxList) = i


# RGB = label2rgb(ii)
# figure, imshow(RGB)

# 把每一条基线连接起来
[cl, ~] = JoinAllBaseLine(ii, graindata, tc)
clp = cell(lineNum, 1)
[h, w] = size(ii)
baseLineImage = zeros(h, w)
for j=1:lineNum
clp
{j} = sub2ind([h, w], int32(cl
{j}(2,:)), int32(cl
{j}(1,:)))
baseLineImage(clp
{j}) = j


dify = zeros(lineNum, w)
for i = 1:lineNum - 1
for j=1:w
x1 = find(cl
{i}(1,:) == j )
y1 = cl
{i}(2, x1(1))

x2 = find(cl
{i + 1}(1,:) == j )
y2 = cl
{i + 1}(2, x2(1))
dify(i, j) = y2 - y1


dify(lineNum,:) = dify(lineNum - 1,:)



function[cl, baseLineImage] = JoinAllBaseLine(ii, graindata, tc)
# 连接所有的基线　返回一个有不同标记的基线图
bbx = [graindata.Extrema]
[h, w] = size(bbx)
bbx = reshape(bbx, h, 2, w / 2)

baseLineImage = ii
[lineNum, w] = size(tc)
for i=1:lineNum
for j=1:w - 1
if tc(i, j + 1) == 0
    break


leftOne = bbx(:,:, tc(i, j))
rightOne = bbx(:,:, tc(i, j + 1))

dx = rightOne(8, 1) - leftOne(3, 1)
dy = rightOne(8, 2) - leftOne(3, 2)
dy = dy / dx

xstart = leftOne(3, 1)
ystart = leftOne(3, 2)

for x=0:dx - 1
baseLineImage(round(ystart + dy * x), round(xstart + x)) = i




[h, w] = size(ii)
resultImg = baseLineImage * 0
cl = cell(lineNum, 1)
for i=1:lineNum
[i_y, i_x] = ind2sub([h, w], find(baseLineImage == i))
# [i_x, idx] = sort(i_x)
# i_y = i_y(idx)
startx = min(i_x)
x = max(i_x)

len = x - startx + 1
newx = zeros(len, 1)
newy = zeros(len, 1)

for idx=1:len
xvalue = startx + idx - 1
newx(idx) = xvalue
tmp = (i_x == xvalue)
newy(idx) = mean(i_y(tmp))

newy = medfilt1(newy, 31)
newy = medfilt1(newy, 31)
resultImg(sub2ind([h, w], int32(newy), int32(newx))) = i

# cl
{i} = [newx'newy']

resultImg(int32(newy(1)), 1: int32(newx(1))) = i
resultImg(int32(newy(len)), int32(newx(len) + 1): w) = i


for i=1:lineNum
[i_y, i_x] = ind2sub([h, w], find(resultImg == i))
cl
{i} = [i_x'i_y']

baseLineImage = resultImg


function[tc] = GetTrackMatrix(dm, maxvalue, graindata, I)
# 追踪所有的水平连通区域
[~, newIdx] = sort(dm, 2, 'asc')

# 追踪每一个单位
[h, w] = size(dm)
tc = zeros(h) # 追踪队列

for curid1 = 1:h
curid2 = curid1
tccount = 1 # 当前追踪队列的序号
while curid2 <= h
    tc(curid1, tccount) = curid2
    tccount = tccount + 1

    # 取下一个单位
    curid3 = curid2
    bget = 0
    while curid3 <= h
        if (curid3 == newIdx(curid3, 1)) # 如果最近的是本身 那么就不取
        break


    curid3 = newIdx(curid3, 1)
    if dm(curid2, curid3) == maxvalue
        bget = 0
    else
        bget = 1

    break


# 如果能取到下一个单位
if bget == 1
    dst = dm(curid2, curid3)
    if dst == maxvalue # 如果下一个单位无法到达也退出
        break

    curid2 = curid3
else # 取不到就退出
break




ss = tc > 0
ssc = sum(ss, 2)
[~, sidx] = sort(ssc, 'desc')
newTc = tc(sidx,:)

for i = 1:h - 1 # 从最长的开始判断
合并所有子集
baseL = newTc(i,:)
if sum(baseL) == 0
    continue

len = sum(baseL > 0)
baseL = baseL(1:len)

for j = i+1:h
subL = newTc(j,:)
if sum(subL) == 0
    continue

len = sum(subL > 0)
subL = subL(1:len)
bsub = intersect(subL, baseL)
if ~isempty(bsub)
    newTc(j,:) = 0




pl = {graindata.PixelIdxList}
pointNum = zeros(1, h)
zk = zeros(1, h) # 空占比
[hh, ww] = size(I)
tmpImg = zeros([hh, ww])
for i=1:h
baseL = newTc(i,:)
if sum(baseL) == 0
    continue


gminx = zeros(1, w)
gmaxx = zeros(1, w)
for j=1:w
qyidx = newTc(i, j) # 获得连通区域编号
if qyidx == 0
    continue


[~, c_x] = ind2sub([hh, ww], pl
{qyidx} )
tmpImg(pl
{qyidx} ) = i
minx = min(c_x)
maxx = max(c_x)
pointNum(i) = pointNum(i) + maxx - minx + 1
gminx(j) = minx
gmaxx(j) = maxx

gminx = gminx(find(gminx > 0))
len = max(gmaxx) - min(gminx) + 1
zk(i) = pointNum(i) / len


# RGB = label2rgb(tmpImg)
# figure, imshow(RGB)
zk = zk
'

ss = newTc > 0
ssc = sum(ss, 2)
t1 = max(ssc) / 3
t2 = 0.3

idx1 = find(ssc > t1)
idx2 = find(zk > t2)
idx = intersect(idx1, idx2)
tc = newTc(idx,:)


function[dm] = GetDistanceMatrix(graindata, maxvalue)
# 计算所有连通区域的距离矩阵

bbx = [graindata.BoundingBox]
[h, w] = size(bbx)
bbx = reshape(bbx, 4, w / 4)
bbx = bbx
'
[h, w] = size(bbx)

dm = zeros(h) # 距离矩阵

for i=1:h
dx = bbx(:, 1)' - (zeros(1,h) + bbx(i,1)+bbx(i,3))
dy = (zeros(1, h) + (bbx(i, 2) + bbx(i, 4)) / 2) - ((bbx(:, 2) + bbx(:, 4)) / 2)'

dy = abs(dy) < 5 # y的差异小于某一阈值
dy = ~dy
dx(dy) = maxvalue
dy = dx < 0 # 必须在当前点右边
dx(dy) = maxvalue
dx(i) = maxvalue # 不计算自身点
dm(i,:) = dx




from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import imageio
from cv2 import cv2


def DiscreteRadonTransform(image, steps):
    channels = len(image[0])
    res = np.zeros((channels, channels), dtype='float64')
    for s in range(steps):
        rotation = ndimage.rotate(image, -s*180/steps, reshape=False).astype('float64')
        #print(sum(rotation).shape)
        res[:,s] = sum(rotation)
    return res

#读取原始图片
#image = cv2.imread("whiteLineModify.png", cv2.IMREAD_GRAYSCALE)
#image=imageio.imread('shepplogan.jpg').astype(np.float64)
#image = cv2.imread("whitePoint.png", cv2.IMREAD_GRAYSCALE)
# image = cv2.imread("./001.18.png", cv2.IMREAD_GRAYSCALE)
# img=cv2.resize(image,(2000,1024))
# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# radon = DiscreteRadonTransform(img, len(img[0]))
# print(radon.shape)
# cv2.imshow('img',radon )
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#绘制原始图像和对应的sinogram图
# plt.subplot(1, 2, 1)
# plt.imshow(image, cmap='gray')
# plt.subplot(1, 2, 2)
# plt.imshow(radon, cmap='gray')
# plt.show()
# -*- coding: UTF-8 -*-

import numpy as np
import cv2


## 图片旋转
def rotate_bound(image, angle):
    # 获取宽高
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # 提取旋转矩阵 sin cos
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 计算图像的新边界尺寸
    # nW=w
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # nH = h

    # 调整旋转矩阵
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


## 获取图片旋转角度
def get_minAreaRect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    return cv2.minAreaRect(coords)


image_path = "006.365.png"
image = cv2.imread(image_path)
image=cv2.resize(image,(2000,1024))
h=image.shape[1]
w=image.shape[0]
angle = get_minAreaRect(image)[-1]#得到旋转角度
rotated = rotate_bound(image, angle)#旋转图像
M = cv2.getRotationMatrix2D((w/2,w/2),-angle,1)#旋转图像中心点，角度，比例
dst = cv2.warpAffine(rotated, M, (h,w))#旋转
# dst=cv2.resize(dst,(2000,1024))
# image22 = cv2.flip(rotated , 1)
#
# cv2.putText(rotated, "angle: {:.2f} ".format(angle),
#             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.putText(rotated, "angle: {:.2f} ".format(angle),
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 0)
cv2.imshow("rotated", rotated)
cv2.imshow("imput", image)
cv2.imshow("output", dst)
cv2.waitKey(0)
# show the output image
print("[INFO] angle: {:.3f}".format(angle))
# cv2.imshow("imput", image)
# cv2.imshow("output", rotated)
# cv2.waitKey(0)
'''