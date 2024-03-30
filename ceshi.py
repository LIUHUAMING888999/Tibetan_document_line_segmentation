
# 导入工具包
from imutils import contours
from scipy import ndimage
import numpy as np
import argparse
import cv2


# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")

args = vars(ap.parse_args())

# 绘图展示
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 读取输入图像，预处理
image = cv2.imread(args["image"])
cv_show('image', image)
imag = cv2.resize(image, (2000, 1024))
img = cv2.imread(args["image"], 0)

# img= DiscreteRadonTransform(img , len(image[0]))

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
dst1 = cv2.erode(gray, k, iterations=5)
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

threshCnts, hierarchy = cv2.findContours(np.uint8(thresh.copy()), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts = threshCnts
print(hierarchy)
cur_img = imag.copy()
cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)
cv_show('img', cur_img)
locs = []
'''
# 遍历轮廓
for (i, c) in enumerate(cnts):
    # 计算矩形
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    # 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
    if ar > 1 and ar < 40000:

        if (w > 800 and w < 2000) and (h > 15 and h < 4000):
            for u in range(8):
                y = int(195*(u+1)*0.5)
                x = 200
                h = 65
                w = 1700
                # 符合的留下来
                locs.append((x, y, w, h))

# 将符合的轮廓从左到右排序
locs = sorted(locs, key=lambda x: x[0])
output = []

# 遍历每一个轮廓中的数字
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    # initialize the list of group digits
    groupOutput = []

    # 根据坐标提取每一个组
    group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    cv_show('group', group)

    # 画出来
    cv2.rectangle(imag, (gX - 5, gY - 5),
                  (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
    cv2.putText(imag, "".join(groupOutput), (gX, gY - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # 得到结果
    output.extend(groupOutput)
'''
# 打印结果

# print("Credit Card #: {}".format("".join(output)))
cv2.imshow("Image", imag)
cv2.waitKey(0)
