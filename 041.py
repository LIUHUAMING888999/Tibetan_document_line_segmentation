#迭代旋转图像
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
# 初始化
a=1
while a<3:
    if a==1:
        root_path = "F:/cc/remove_border/"
    else:
        root_path = 'F:/ccrenew/' + "0357png7" + "/" +"png%d"%(a-1)+"/"
    dir = root_path
    count = 0
    for root, dir, files in os.walk(dir):
        for file in files:
            bw2 = getCorrect(root_path + str(file))
            cv2.imwrite('F:/ccrenew/' + "0357png7" + "/" +"png%d"%(a)+"/"+ str(file), bw2)
            count += 1
            print(count)
    a+=1
