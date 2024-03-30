'''
批处理
'''
import cv2
import os

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
root_path = "F:/ccrenew/0354png5/"
dir = root_path
count = 0
for root, dir, files in os.walk(dir):
    for file in files:
        img = cv2.imread(root_path + str(file))
        cv_show('ds',img)
        count+=1
        print(count)