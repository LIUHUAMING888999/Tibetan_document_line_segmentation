# '''
# 轮廓检测，剔除上元因
# '''
# from imutils import contours
# import numpy as np
# import argparse
# import cv2 as cv
# import myutils
# def cv_show(name,image):
#     cv.imshow(name,image)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
#
#
# def findcounts(image):
#     # image = cv.imread(image)
#     sqKernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 7))
#     gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#     ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
#     binary=cv.erode(binary,sqKernel)
#     image_c=image.copy()
#     threshCnts, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#     cv.drawContours(image_c, threshCnts, -1, (0, 0, 255), 1)
#     cv_show('j', image_c)
#     locs = []
#
#     # 遍历轮廓
#     for (i, c) in enumerate(threshCnts):
#         # 计算矩形
#         (x, y, w, h) = cv.boundingRect(c)
#         ar = w / float(h)
#
#         # 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
#         if ar > 0.5and ar < 1.5:
#
#             if (w > 5and w < 50) and (h >17and h <50):
#                 # 符合的留下来
#                 locs.append((x, y, w, h))
#
#     # 将符合的轮廓从左到右排序
#     locs = sorted(locs, key=lambda x: x[0])
#     output = []
#
#     # 遍历每一个轮廓中的数字
#     for (i, (gX, gY, gW, gH)) in enumerate(locs):
#         # initialize the list of group digits
#         groupOutput = []
#
#         # 根据坐标提取每一个组
#         # group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
#         group = gray[gY :gY + gH , gX :gX + gW ]
#         # cv_show('group', group)
#         # 预处理
#         group = cv.threshold(group, 0, 255,
#                               cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
#         # cv_show('group', group)
#         # 计算每一组的轮廓
#         digitCnts, hierarchy = cv.findContours(group.copy(), cv.RETR_EXTERNAL,
#                                                       cv.CHAIN_APPROX_SIMPLE)
#         digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
#
#         # 计算每一组中的每一个数值
#         for c in digitCnts:
#             # 找到当前数值的轮廓，resize成合适的的大小
#             (x, y, w, h) = cv.boundingRect(c)
#             roi = group[ y:y + h,x:x + w]
#             # roi = cv.resize(roi, (200, 100))
#             # cv_show('roi', roi)
#
#             # 计算匹配得分
#             scores = []
#
#             # # 在模板中计算每一个得分
#             # for (digit, digitROI) in digits.items():
#             #     # 模板匹配
#             #     result = cv.matchTemplate(roi, digitROI,
#             #                                cv.TM_CCOEFF)
#             #     (_, score, _, _) = cv.minMaxLoc(result)
#             #     scores.append(score)
#             #
#             # # 得到最合适的数字
#             # groupOutput.append(str(np.argmax(scores)))
#
#         # 画出来
#         cv.rectangle(image, (gX, gY ),
#                       (gX + gW , gY + gH), (0, 0, 255), 1)
#         cv.putText(image, "".join(groupOutput), (gX, gY - 15),
#                     cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
#
#         # 得到结果
#         output.extend(groupOutput)
#
#     # 打印结果
#
#     print("Credit Card #: {}".format("".join(output)))
#
#     cv.imshow("image", image)
#     cv.waitKey(0)
#     return len(locs)
# imaa=cv.imread('dashen_compressed.png')
# findcounts(imaa)

import cv2
def findcounts(image):
    # image = cv2.imread('dashen_compressed.png')
    image=cv2.resize(image,(2000,1024))
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(image1, 127, 255, cv2.THRESH_BINARY)
    contours, hierarch = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image, contours, -1, (0, 0, 255), 1)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 60 and area < 400:
        # if area<=50:
            # cv2.drawContours(image, [contours[i]], -1, (0, 0, 255), 1)
            cv2.fillPoly(image, [contours[i]], (255, 255, 255))
    # imagess = cv2.imread('001.18.png')
    # imagess=cv2.resize(imagess,(2000,1024))
    # image=cv2.bitwise_not(image)
    return image
    # imageg = cv2.bitwise_and(image, imagess)
    # cv2.imshow('name', imageg)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
# def cv_show(name, img):
#     cv2.imshow(name, img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# asd=cv2.imread('001.18.0.png')
# # asd=cv2.bitwise_not(asd)
# asda=cv2.imread('dashen_compressed.png')
# uia=findcounts(asda)
# cv_show('am',uia)
# ahj=cv2.bitwise_xor(asd,uia)
# ahj=cv2.bitwise_not(ahj)
# cv_show('ajskd',ahj)