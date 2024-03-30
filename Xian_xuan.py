import math
'''
输入坐标，中心点，角度得到旋转后坐标
'''

def onepoint(x, y, n, m,angle):
    angle=angle*math.pi/180#(角度 = 180 / π * 弧度)将角度转为弧度
    # X = x*math.cos(angle) - y*math.sin(angle)-0.5*n*math.cos(angle)+0.5*m*math.sin(angle)+0.5*n
    # Y = y*math.cos(angle) + x*math.sin(angle)-0.5*n*math.sin(angle)-0.5*m*math.cos(angle)+0.5*m
    X = x * math.cos(angle) - y * math.sin(angle) - n * math.cos(angle) + m * math.sin(angle) + n
    Y = y * math.cos(angle) + x * math.sin(angle) - n * math.sin(angle) - m * math.cos(angle) + m
    # return [round(X), int(Y)]
   #1 return [int(X), int(Y)]
    return [int(X), y]


# newrect = []
# newrect = onepoint(2, 2, 0, 0,90)
# print(newrect)
