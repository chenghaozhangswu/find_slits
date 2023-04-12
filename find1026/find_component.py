# -*- coding: utf-8 -*-
"""
@Time ： 2022/10/21 23:25
@Auth ： zch171041@outlook.com
@File ：find_component.py
@IDE ：PyCharm
@python version:3.6
@Motto：ABC(Always Be Coding)
"""
# -*- coding: utf-8 -*-

from scipy.signal import find_peaks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2


def componemt_dist(data):
    # 输入二维高度numpy数组
    # 输出是排序后的器件距离，第一列数据是索引行，第二列数据是缝隙起始坐标，第三列数据是终止坐标，第四列数据是长度，单位为点
    indx = np.zeros((data.shape[0],4))
    for i in range(data.shape[0]):#data.shape[0]
        # print(i)
        a = data[i, :] 
        a[:][a<-50] = 0
        w = 5  # 采用频率为10的时候w=5 
        a_var = np.zeros_like(a)
        for j in range(w+1,len(a)-w):
            a_var[j] = np.var(a[j-w:j+w])
        # plt.plot(a_var)
        # plt.plot(a)

        location,_ = find_peaks(a_var, height=0.017,distance=3)
        # for k in range(len(location)):
        #     plt.axvline(x=location[k], ls='--', c='red')
        # plt.show()
        temp = []
        distance = []
        if len(location)>1:
            for k in range(1, len(location)-1):
                q = np.mean(a[location[k - 1]:location[k]])
                h = np.mean(a[location[k]:location[k + 1]])
                if q-h>0.2:
                    temp = np.append(temp,k)
                    distance = np.append(distance,location[k + 1] - location[k])
            if len(temp)!=0:
                n = np.argmin(np.array(distance)) #
                indx[i, 0] = i
                indx[i, 1] = location[int(temp[n])]
                indx[i, 2] = location[int(temp[n])+1]
                indx[i, 3] = distance[n]

    #indx = indx[[not np.all(indx[i] == 0) for i in range(indx.shape[0])], :]
    #indx = np.array(sorted(indx, key=lambda x: x[3]))
    return indx