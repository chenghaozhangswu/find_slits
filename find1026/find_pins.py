# -*- coding: utf-8 -*-
"""
@Time ： 2022/10/21 23:32
@Auth ： zch171041@outlook.com
@File ：find_pins.py
@IDE ：PyCharm
@python version:3.6
@Motto：ABC(Always Be Coding)
"""

from scipy.signal import find_peaks
from scipy.signal import medfilt
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt


def pins_dist(data):
    # 输入二维高度numpy数组
    # 输出是平均引脚距离，第一列是索引也是横坐标，第二列是引脚中间坐标，第三列是平均间距
    result = np.zeros((data.shape[0], 5))
    for j in range(0,data.shape[0],3):
        a = data[j, :]#+3.8 # complement
        al = len(a)

        # delete Singular value
        for i in range(al):
            if a[i]<-50:
                a[i] = 0


        # calculate variance
        a_var = np.zeros_like(a)

        w=2
        for i in range(w+1, al-w):
            a_var[i] = np.var(a[i-w:i+w])
        # plt.plot(a_var)
        # plt.show()
        # set var range
        for i in range(al):
            if a_var[i]>0.001*5:
                a_var[i] = 0


        a_var1 = copy.copy(a_var)
        for i in range(al):
            if a_var1[i]>0.09*0.001 and a_var1[i]<3*0.001:
                a_var1[i] = 0.3*0.001

        # median filtering
        a_var1 = 10000*medfilt(a_var1, 9)
        # plt.plot(a)
        # # plt.plot(a_var1)
        # plt.show()
        qian = 0
        hou = 0
        threshold = 100
        for i in range(1, al-threshold):
            if a_var1[i] == np.mean(a_var1[i:i + threshold]) and round(a_var1[i])==3:
                qian = i
                for k in range(i+threshold, al):
                    if round(a_var1[k])!=3:
                        hou = k
                        break
                break
        xian = a_var[qian:hou]
        a = a[qian:hou]
        # plt.plot(a_var[qian:hou])
        # plt.show()
        location, _ = find_peaks(xian, height=0, distance=1)
        q = []
        h = []
        if len(location)>1:
            for i in range(len(location)):
                if a[location[i]-1] > a[location[i]]:
                    h = np.append(h, i)
                else:
                    q = np.append(q, i)
            distance = []
            for i in range(len(h)-1):
                distance = np.append(distance, location[int(h[i])+1]-location[int(h[i])])
            # if qian != 0 and np.mean(a_var[qian:hou]) < 0.0006:
            #     distance = []
            if len(distance)>0:
                md = np.mean(np.array(distance))
                result[j,0] = j
                #result[j,1] = int((qian+hou)/2)
                result[j,1] = qian
                result[j,2] = hou

                result[j,3] = md
                result[j,4] = len(distance)+1 # len(distance)是间距的个数  len(distance)+1引脚的个数

    result = result[[not np.all(result[i] == 0) for i in range(result.shape[0])], :]
    return result