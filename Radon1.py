
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


# image_path = "006.365.png"
def xuan_z(image_path):
    image = cv2.imread(image_path)
    # image=255-image
    image=cv2.resize(image,(2000,1024))
    h=image.shape[1]
    w=image.shape[0]
    angle = get_minAreaRect(image)[-1]#得到旋转角度
    rotated = rotate_bound(image, angle)#旋转图像
    if angle>85:
        angle=90
    else:
        angle=angle
    M = cv2.getRotationMatrix2D((w/2,w/2),angle,1)#旋转图像中心点，角度，比例
    dst = cv2.warpAffine(rotated, M, (h,w))#旋转

    cv2.putText(rotated, "angle: {:.2f} ".format(angle),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # cv2.imshow("rotated", rotated)
    # cv2.imshow("imput", image)
    # cv2.imshow("output", dst)
    # cv2.waitKey(0)
    print("[INFO] angle: {:.3f}".format(angle))
    return dst
