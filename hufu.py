import numpy as np
import os
import cv2
import math
# from scipy import misc, ndimage
# @@@@@@@@@@@
# 使用霍夫变换
##########
def rotate(image, angle, center=None, scale=1.0):
    (w, h) = image.shape[0:2]
    if center is None:
        center = (w // 2, h // 2)
    wrapMat = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, wrapMat, (h, w))


# 使用霍夫变换
def getCorrect2():
    # 读取图片，灰度化
    src = cv2.imread("019.515.png", cv2.IMREAD_COLOR)
    src=cv2.resize(src,(2000,1024))
    showAndWaitKey("src", src)
    gray = cv2.imread("019.515.png", cv2.IMREAD_GRAYSCALE)
    gray= cv2.resize(gray, (2000, 1024))
    showAndWaitKey("gray", gray)
    # 腐蚀、膨胀
    kernel = np.ones((3,3), np.uint8)
    erode_Img = cv2.erode(gray, kernel)
    eroDil = cv2.dilate(erode_Img, kernel)
    showAndWaitKey("eroDil", eroDil)
    # 边缘检测
    canny = cv2.Canny(eroDil, 50, 150)
    showAndWaitKey("canny", canny)
    # 霍夫变换得到线条
    lines = cv2.HoughLinesP(canny, 0.8, np.pi / 180, 90, minLineLength=100, maxLineGap=10)
    drawing = np.zeros(src.shape[:], dtype=np.uint8)
    # 画出线条
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(drawing, (x1, y1), (x2, y2), (0, 255, 0), 1, lineType=cv2.LINE_AA)

    showAndWaitKey("houghP", drawing)
    """
    计算角度,因为x轴向右，y轴向下，所有计算的斜率是常规下斜率的相反数，我们就用这个斜率（旋转角度）进行旋转
    """
    k = float(y1 - y2) / (x1 - x2)
    thera = np.degrees(math.atan(k))
    print(thera)

    """
    旋转角度大于0，则逆时针旋转，否则顺时针旋转
    """
    rotateImg = rotate(src, thera)
    cv2.imshow("rotateImg", rotateImg)
    cv2.waitKey()


def showAndWaitKey(winName, img):
    cv2.imshow(winName, img)
    cv2.waitKey()


if __name__ == "__main__":
    getCorrect2()
