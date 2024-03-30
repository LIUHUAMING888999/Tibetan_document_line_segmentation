
# 批处理
# 根据分块行投影求出各区域近似基线
# p0355png6  上下对应基线不能相差过大
# 上下对应基线不能差距过大
# 与0357和0358相比，进行基线前后对照的二值图改变
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
from untitled0 import *
from Xuan_zhuan_c import *
from Collect_fenx import *
import matplotlib.pyplot as plt


def Jiu_zheng1(a, b):  # a起点位置b结束位置
    for X1_X2_X3 in np.arange(a, -1, -1):  # 选取11段基线处为基准，向前纠正
        if (abs(Collect1_x[X1_X2_X3] - Collect1_x[
            X1_X2_X3 - 2]) > 10 or abs(
            Collect1_x[(X1_X2_X3 + 1)] - Collect1_x[
                (X1_X2_X3 - 1)]) > 10):
            Collect1_x[(X1_X2_X3 - 2)] = Collect1_x[X1_X2_X3]
            Collect1_x[(X1_X2_X3 - 1)] = Collect1_x[X1_X2_X3 + 1]
    for X1_X2_X3 in np.arange(a + 1, b, 1):  # 选取12段基线处为基准，向后纠正
        if (abs(Collect1_x[X1_X2_X3] - Collect1_x[
            X1_X2_X3 + 2]) > 10 or abs(
            Collect1_x[(X1_X2_X3 + 1)] - Collect1_x[
                (X1_X2_X3 + 3)]) > 10):
            Collect1_x[(X1_X2_X3 + 2)] = Collect1_x[X1_X2_X3]
            Collect1_x[(X1_X2_X3 + 3)] = Collect1_x[X1_X2_X3 + 1]


def Jiu_zheng(a, b):  # a起点位置b结束位置,纠正
    for ui in range(len(peaks_b)):
        for X1_X2_X3 in np.arange(a, -1, -1):  # 选取11段基线处为基准，向前纠正
            if (abs(Collect_up_down_x[ui][X1_X2_X3] - Collect_up_down_x[ui][
                X1_X2_X3 - 2]) > 8 or abs(
                Collect_up_down_x[ui][(X1_X2_X3 - 1)] - Collect_up_down_x[ui][
                    (X1_X2_X3 - 3)]) > 8):
                Collect_up_down_x[ui][(X1_X2_X3 - 2)] = Collect_up_down_x[ui][X1_X2_X3]
                Collect_up_down_x[ui][(X1_X2_X3 - 1)] = Collect_up_down_x[ui][X1_X2_X3 + 1]
        for X1_X2_X3 in np.arange(a + 2, b, 1):   # 选取12段基线处为基准，向后纠正
            if (abs(Collect_up_down_x[ui][X1_X2_X3] - Collect_up_down_x[ui][
                X1_X2_X3 + 2]) > 8 or abs(
                Collect_up_down_x[ui][(X1_X2_X3 + 1)] - Collect_up_down_x[ui][
                    (X1_X2_X3 + 3)]) > 8):
                Collect_up_down_x[ui][(X1_X2_X3 + 2)] = Collect_up_down_x[ui][X1_X2_X3]
                Collect_up_down_x[ui][(X1_X2_X3 + 3)] = Collect_up_down_x[ui][X1_X2_X3 + 1]


def Collect_line1():
    # cv2.line(img_c, (YYY1, hang - 1), (YYY2, hang - 1), (0, 0, 255), 1)
    Collect1_x.append(hang - 1)
    Collect1_x.append(hang - 1)
    Collect1_y.append(YYY1)
    Collect1_y.append(YYY2)


def Collect_line2():
    # cv2.line(img_c, (YYY1, XXX1), (YYY2, XXX2), (0, 0, 255), 1)
    Collect1_x.append(XXX1)
    Collect1_x.append(XXX2)
    Collect1_y.append(YYY1)
    Collect1_y.append(YYY2)


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 设置参数
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
#                 help="path to input image")
#
# args = vars(ap.parse_args())

# 初始化
a_diedai = 1
while a_diedai < 5:
    if a_diedai == 1:
        root_path = "F:/cc/remove_border1/"
    else:
        root_path = 'F:/ccrenew/' + "03510png10" + "/" + "png%d" % (a_diedai +a_diedai- 3) + "/"
    dir = root_path
    count = 0
    for root, dir, files in os.walk(dir):
        for file in files:
            img = cv2.imread(root_path+str(file))
            img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # cv_show('img', img1)
            img2 = cv2.resize(img1, (2000, 1024))
            # cv_show('img', img2)
            # 反向二值

            bw2 = getCorrect(root_path+str(file))  # 切片专用
            #
            # print(bw2.shape)
            # cv_show('bw', bw2)

            bw = bw2 / 255

            bw1 = 1 - bw
            bw3 = 1 - bw1
            # cv_show('iimgbw1', bw1)

            width = bw3.shape[0]
            height = bw3.shape[1]
            # print(height, width)
            point_w = []
            point_b = []
            # 统计每行像素值
            for row in range(width):
                a = 0
                b = 0
                # if row < 100 or row > 900:
                #     val = bw3[50][50]
                for col in range(height):
                    # if col < 100 or col > 1900:
                    #     val = bw3[50][50]
                    # else:
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
                # point_w.append(a)
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
            # plt.plot(y_b, 'b', label='savgol')
            # plt.title('feng')
            # plt.subplot(212)
            # plt.plot(y_w, 'b', label='savgol')
            # plt.title('gu')
            # plt.show()

            # 输出各峰谷，峰值所在行
            # x_w = point_w[0:4000]
            # peaks_w, _ = find_peaks(x_w, height=1200, distance=65)
            x_b = point_b[0:4000]
            peaks_b, _ = find_peaks(x_b, height=500, distance=55)
            # print(peaks_b)
            # print(peaks_w)

            # 根据波峰波谷进行切片
            img_crop = bw2.copy()

            img_c = bw2.copy()
            img_connect = bw2.copy()
            # c=int(height-(0.125)*height)
            # d=int(height-(0.875)*height)
            # print(c,d)
            # img_crop1=img_c[205:215,d:c]
            # cv_show('igh',img_crop1)
            num_labels, centroids = Collect_fenxi(img_connect)  # 连通域中心点
            a = int(height - (0.125) * height)  # 切到八分之七行
            b = int(height - (0.875) * height)  # 切到八分之一行
            c = int(height - (0.055) * height)  # 切到行尾
            d = int(height - (0.95) * height)  # 切到行头
            e = int((c - d) / 40)  # 分成三十份后，一份的宽
            Collect_up_down_x = []  # 收集所有基线横坐标，上下对照用
            Collect_up_down_y = []  # 收集所有基线横坐标，上下对照用

            for i in range(len(peaks_b)):

                # A = peaks_w[i]
                # B = peaks_w[i + 1]
                # try:
                C = peaks_b[i]
                # except IndexError:
                #     print('未检测8个峰')
                # if abs(C - A) <= 10:
                #     A -= 20
                # if B - A > 120:
                #     B -= 9
                cv2.line(img_c, (0, C), (2000, C), (0, 255, 0), 1)
                # imag=img_crop[A:B,:]
                # cv_show('img',imag)
                # A = peaks_w[0]
                # B=(peaks_w[-1]-A)//8#平均值
                # imag=img_crop[(A+B*i):(A+B*(i+1)),:]
                # cv_show('img',imag)
                for hang in range(width):
                    C_a = 0
                    if hang == C:
                        Collect1_x = []  # 收集基线端点横坐标，连线用
                        Collect1_y = []  # 收集基线端点纵坐标，连线用
                        for Collect_x in range(num_labels):  # 统计波峰穿过连通域个数，超过3个就极大削弱约束1
                            if abs(centroids[Collect_x][1] - C) <= 3:
                                C_a += 1
                        # print(C_a)
                        for h in range(40):
                            f = d + e * (h + 1)
                            g = d + e * h
                            # print(c)
                            # print(d)
                            # print(e)
                            # print(f)
                            # img_crop1 = img_c[hang - 12:hang + 7, g:f]  # 切割波峰附近区域

                            if C_a >= 3:  # 统计波峰穿过连通域个数，超过3个就极大削弱约束1
                                if g < b or g > a:
                                    img_crop1 = img_c[hang - 30:hang + 7, g:f]  # 切割波峰附近区域
                                else:
                                    img_crop1 = img_c[hang - 40:hang + 7, g:f]  # 切割波峰附近区域
                                T_m = []  # 收集一张图片的【最大像素所在行：像素值】
                                T_t_a = []  # 收集每个角度的峰值
                                T_t_angle = []  # 收集每个峰值对应的角度，以便找出最大峰值所对应的角度
                                Collect_xy_start = []  # 收集旋转后的基线起点坐标
                                Collect_xy_end = []  # 收集旋转后的基点结尾坐标
                                for angle1 in np.arange(-5, 5.5, 0.5):  # 从（-5，5）步长为0.5
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
                                g_f = (g + f) / 2
                                if g < b or g > a:  # 行头行尾部分
                                    XX = int(hang - (31 - find_hang) - 2)  # 局部行所对应的全局行
                                else:
                                    XX = int(hang - (41 - find_hang) - 2)  # 局部行所对应的全局行
                                # print(XX)
                                # print(find_height)
                                # print(g_f)
                                # print(find_angle)
                                Collect_xy_start = onepoint(XX, g, find_hang, g_f, find_angle)
                                # print(Collect_xy_start)
                                # print(Collect_xy_start)
                                find_angle = find_angle * math.pi / 180
                                # if find_angle>=0 :
                                XXX1 = Collect_xy_start[0]
                                # XXX1=int(XXX1+int(50*math.tan(-find_angle)))
                                YYY1 = Collect_xy_start[1]
                                # YYY1=round(YYY1-math.tan(find_angle))
                                Collect_xy_end = onepoint(XX, f, find_hang, g_f, find_angle)
                                # print(Collect_xy_end)
                                XXX2 = Collect_xy_end[0]
                                YYY2 = Collect_xy_end[1]
                                # YYY2=round(YYY2-math.tan(find_angle))
                                # print(XX)
                                # print(find_hang)
                                # print(find_height)
                                # cv2.line(img_c, (g, XX), (f, XX), (0, 0, 255), 1)#各基线未旋转前
                                if g < b or g > a:  # 在行头和行尾部分减小约束
                                    if abs(hang - XXX1) > 35:
                                        Collect_line1()
                                    else:
                                        Collect_line2()
                                else:
                                    if abs(hang - XXX1) > 40:
                                        Collect_line1()
                                    else:
                                        Collect_line2()
                            else:
                                if g < b or g > a:
                                    img_crop1 = img_c[hang - 30:hang +15, g:f]  # 切割波峰附近区域
                                else:
                                    img_crop1 = img_c[hang - 20:hang + 7, g:f]  # 切割波峰附近区域
                                T_m = []  # 收集一张图片的【最大像素所在行：像素值】
                                T_t_a = []  # 收集每个角度的峰值
                                T_t_angle = []  # 收集每个峰值对应的角度，以便找出最大峰值所对应的角度
                                Collect_xy_start = []  # 收集旋转后的基线起点坐标
                                Collect_xy_end = []  # 收集旋转后的基点结尾坐标
                                for angle1 in np.arange(-5, 5.5, 0.5):  # 从（-5，5）步长为0.5
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
                                g_f = (g + f) / 2
                                if g < b or g > a:  # 行头行尾部分
                                    XX = int(hang - (31 - find_hang) - 2)  # 局部行所对应的全局行
                                else:
                                    XX = int(hang - (21 - find_hang) - 2)  # 局部行所对应的全局行

                                # print(XX)
                                # print(find_height)
                                # print(g_f)
                                # print(find_angle)
                                Collect_xy_start = onepoint(XX, g, find_hang, g_f, find_angle)
                                # print(Collect_xy_start)
                                # print(Collect_xy_start)
                                find_angle = find_angle * math.pi / 180

                                # if find_angle>=0 :
                                XXX1 = Collect_xy_start[0]
                                # XXX1=int(XXX1+int(50*math.tan(-find_angle)))
                                YYY1 = Collect_xy_start[1]
                                # YYY1=round(YYY1-math.tan(find_angle))
                                Collect_xy_end = onepoint(XX, f, find_hang, g_f, find_angle)
                                # print(Collect_xy_end)
                                XXX2 = Collect_xy_end[0]
                                YYY2 = Collect_xy_end[1]
                                # print(find_hang)
                                # print(find_height)
                                # cv2.line(img_c, (g, XX), (f, XX), (0, 0, 255), 1)#各基线未旋转前
                                if g < b or g > a:  # 在行头和行尾部分减小约束
                                    if abs(hang - XXX1) > 30:
                                        Collect_line1()
                                    else:
                                        Collect_line2()
                                else:
                                    if abs(hang - XXX1) > 15:
                                        Collect_line1()
                                    else:
                                        Collect_line2()
                        # print(len(Collect1_x))
                        # for x_y in range(1, 80):
                        for X1_X2_X3 in range(1, 41):  # 中间位置，每块区域基线与前后两条对照差距不能过大
                            if (abs(Collect1_x[X1_X2_X3 + (X1_X2_X3 - 2)] - Collect1_x[
                                X1_X2_X3 + (X1_X2_X3 - 4)]) > 3 and abs(
                                Collect1_x[X1_X2_X3 + (X1_X2_X3 - 1)] - Collect1_x[
                                    X1_X2_X3 + (X1_X2_X3 - 3)]) > 3 and abs(
                                Collect1_x[X1_X2_X3 + (X1_X2_X3)] - Collect1_x[X1_X2_X3 + (X1_X2_X3 - 2)]) > 3 and abs(
                                Collect1_x[X1_X2_X3 + (X1_X2_X3 + 1)] - Collect1_x[X1_X2_X3 + (X1_X2_X3 - 1)]) > 3):
                                Collect1_x[X1_X2_X3 + (X1_X2_X3 - 2)] = Collect1_x[X1_X2_X3 + (X1_X2_X3)]
                                Collect1_x[X1_X2_X3 + (X1_X2_X3 - 1)] = Collect1_x[X1_X2_X3 + (X1_X2_X3) + 1]
                            if abs(Collect1_x[-3] - Collect1_x[-1]) > 3 and abs(
                                    Collect1_x[-4] - Collect1_x[-2]) > 3:
                                Collect1_x[-1] = Collect1_x[-5]
                                Collect1_x[-2] = Collect1_x[-6]
                        # Jiu_zheng1(28, 77)  # 以第12段基线前后对照
                        # Jiu_zheng1(75, 77)  # 取行尾基线往前纠正
                        Collect_up_down_x.append(Collect1_x)
                        Collect_up_down_y.append(Collect1_y)
            # print((Collect_up_down_x))
            # print((Collect_up_down_y))
            if count < 180:
                for x_up_down in range(len(peaks_b) - 1):
                    for C_x_y in range(79):
                        if Collect_up_down_x[x_up_down + 1][C_x_y] - Collect_up_down_x[x_up_down][C_x_y] < 70:
                            Collect_up_down_x[x_up_down][C_x_y] = Collect_up_down_x[x_up_down][C_x_y] - 20
                        elif Collect_up_down_x[x_up_down + 1][C_x_y] - Collect_up_down_x[x_up_down][C_x_y] > 120:
                            Collect_up_down_x[x_up_down][C_x_y] = Collect_up_down_x[x_up_down][C_x_y] + 10

            Jiu_zheng(28, 77)  # 以第12段基线前后对照
            Jiu_zheng(74, 77)  # 取行尾基线往前纠正
            for x_y in range(len(peaks_b)):
                for X_Y in range(79):
                    cv2.line(img_c, (Collect_up_down_y[x_y][X_Y], Collect_up_down_x[(x_y)][X_Y]),
                             (Collect_up_down_y[(x_y)][X_Y + 1], Collect_up_down_x[(x_y)][X_Y + 1]), (0, 0, 255), 1)
                    # cv2.line(img_c, (Collect1_y[(20)], 0),
                    #          (Collect1_y[(20)],1000), (255, 0, 0), 1)

            # cv_show('opi', img_c)
            cv2.imwrite('F:/ccrenew/' + "03510png10" + "/" + "png%d" % (a_diedai*2-1) + "/" + str(file), bw2)
            cv2.imwrite('F:/ccrenew/' + "03510png10" + "/" + "png%d" % (a_diedai*2) + "/" + str(file), img_c)
            count += 1
            # if count % 400 == 0:
            print(count)
    a_diedai += 1
    print(f'第{a_diedai}次迭代')
