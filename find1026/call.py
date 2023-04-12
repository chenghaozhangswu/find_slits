# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 14:19:21 2022

@author: win10
"""

import pandas as pd
import numpy as np
import time
#import find_component
#import find_pins
import cv2
import os
import numpy as np
from skimage import io
import colorsys
import copy
from scipy.signal import medfilt, find_peaks

import cv2
from PIL import Image, ImageDraw, ImageFont
import find_pins
import find_component



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


# 为image添加文字
def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)): # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 字体的格式
        fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
        # 绘制文本
        draw.text((left, top), text, textColor, font=fontStyle)
        # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)




# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

def componemt_dist(data,sample_para_w):
    # 输入二维高度numpy数组
    # 输出是排序后的器件距离，第一列数据是索引行，第二列数据是缝隙起始坐标，第三列数据是终止坐标，第四列数据是长度，单位为点
    indx = np.zeros((data.shape[0], 4))
    for i in range(data.shape[0]):  # data.shape[0]
        # print(i)
        a = data[i, :] 
        a[:][a<-50] = 0
        w = 5  # 采用频率为10的时候w=5 
        a_var = np.zeros_like(a)
        for j in range(w+1,len(a)-w):
            a_var[j] = np.var(a[j-w:j+w])
        # plt.plot(a_var)
        # plt.plot(a)
        horizontal_height = -3.8
        location,_ = find_peaks(a_var, height=0.017,distance=3)
        # for k in range(len(location)):
        #     plt.axvline(x=location[k], ls='--', c='red')
        # plt.show()
        temp = []
        distance = []
        if len(location)>1:
            for k in range(1, len(location)-2):
                q = np.mean(a[location[k - 1]:location[k]])
                h = np.mean(a[location[k]:location[k + 1]])
                hh = np.mean(a[location[k+1]:location[k + 2]])
                if q-h>0.2 and hh-h>0.2 and h>-7.6:
                #if q-h>0.2 and hh-h>0.2 and np.abs(h-horizontal_height)<0.2:
                    temp = np.append(temp, k)
                    distance = np.append(distance, location[k + 1] - location[k])
            if len(temp)!=0:
                n = np.argmin(np.array(distance))
                indx[i, 0] = i
                indx[i, 1] = location[int(temp[n])]
                indx[i, 2] = location[int(temp[n])+1]
                indx[i, 3] = distance[n]  # 原来这个distance是根据采样频率10计算得到的

    #indx = indx[[not np.all(indx[i] == 0) for i in range(indx.shape[0])], :]
    #indx = np.array(sorted(indx, key=lambda x: x[3]))
    return indx


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#



# 获取路径文件夹下面的文件的全部路径
def getlist(filepath):
    name = []
    for file in os.listdir(filepath):
        name.append(filepath+file)
    return name

# list1 = getlist(filepath1)
# list2 = getlist(filepath2)



def downs1(file):
    # 该程序用来读取基恩士原始tif文件并降采样
    # file为指定路径下的tif文件
    # 输出为numpy
    dataframe = io.imread(file)
    sample_para = 10#
    pcd = dataframe[0::sample_para, 0::sample_para]
    pcd = (pcd - 32768.0) * 0.8 * 0.001
    return np.array(pcd)

def hsv2rgb(h, s, v):
    a = list(tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v)))
    r = []
    for i in a:
        r.append(i / 255.0)
    return r

def norms(arr):
    # 归一化函数，arr为需要归一化的数组
    a_max = np.max(arr[np.nonzero(arr)])
    a_min = np.min(arr[np.nonzero(arr)])
    delta = a_max - a_min
    arr_norm = (arr - a_min) / delta
    return arr_norm  # 返回归一化后的数组



def numpy2rgb(height, luminance):
    pcddata = []
    #dataframe = arr1#高度数据
    #v = arr2#明暗数据
    s = 0.5#饱和度--固定值
    
    # 小于水平面的高度全部归为水平面
    #pcd = np.maximum(height, -5)
    
    pcd=height
    #创建一个数组用来保存hsv数据
    
    color_hsv = np.zeros([pcd.shape[0], pcd.shape[1], 3])
    #归一化
    pcdn = norms(pcd)
    #归一化
    vn = norms(luminance)
    
    color_rgb = np.zeros_like(color_hsv)
    
    
    
    for i in range(0,pcd.shape[0]):
        for j in range(0,pcd.shape[1]):
            pcddata.append([i, j,100*pcd[i, j]])
            #pcddata[i, j, 0:3]=[i,j,pcd[i, j]]
            r_g_b=colorsys.hsv_to_rgb(pcdn[i, j], s,vn[i, j])
            r=r_g_b[0]
            g=r_g_b[1]
            b=r_g_b[2]
            color_rgb[i, j, 0:3]=[r, g, b]
            #color_rgb.append([int(i/sample_freq),int(j/sample_freq),[r,g,b]])
            
    print('生成color_rgb成功！！！')
    return color_rgb*255
    # 3d转2d 图片        
    #cv2.imwrite('D:/120.jpg', color_rgb*255)
    




def readpf(string1, string2):
    # 输入tif文件路径 string1表示高度路径 string2表示亮度路径
    # 分别读取高度和亮度
    height_list = getlist(string1)
    luminance_list = getlist(string2)
    
    #list()
    # 从第一个图开始拼接，逆序，所以先处理最后一张图
    #target = numpy2rgb(downs1(height_list[0]), downs(luminance_list[0]))
    #cv2.imwrite('11.jpg', target)
    #source = numpy2rgb(downs1(height_list[1]), downs(luminance_list[1]))
    #cv2.imwrite('22.jpg', source)
    #result = opencvpj.img_stitching(source, target)
    #cv2.imwrite('picture_joint.jpg', result)
    
    list_to_cat=[]
    for i in range(len(luminance_list)):
        list_to_cat.append(numpy2rgb(downs1(height_list[i]), downs(luminance_list[i])))
        
    result = cv2.hconcat(list(reversed(list_to_cat)))

    return result



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

# 获取luminance文件夹内的所有图片
def getlist(filepath):
    name = []
    for file in os.listdir(filepath):
        name.append(filepath + file)
    return name


# 数据降采样
def downs(file, sample_para_h,sample_para_w):
    # 该程序用来读取基恩士原始csv文件并降采样
    # file为指定路径下的csv文件
    # 输出为numpy
    # dataframe = pd.read_csv(file).values
    dataframe = io.imread(file)
    pcd = dataframe[0::sample_para_h, 0::sample_para_w]
    # pcd = np.maximum(pcd, -3.801)
    return np.array(pcd)


# 将原始的多个tiff 拼成  oh,ov
def Tiff2_oh_ov(string1, string2, sample_para_h,sample_para_w):
    filepath1 = string1
    filepath2 = string2

    # 获取路径文件夹下面的文件的全部路径

    list1 = getlist(filepath1)
    list2 = getlist(filepath2)
    num = len(list1)
    
    # 按照 3height  2height  1height 的顺序排列
    height_tiff=list(reversed(list1))
    lunimance_tiff=list(reversed(list2))
    
    
    #height_tiff=list1
    #lunimance_tiff=list2
    
    
    oh=downs(height_tiff[0], sample_para_h,sample_para_w)
    ov=downs(lunimance_tiff[0], sample_para_h,sample_para_w)
    
    print(list1)
    for l in height_tiff[1:]:
        print(l)
        temp_oh = downs(l, sample_para_h,sample_para_w)
        oh = np.hstack((oh, temp_oh))
    
    for l in lunimance_tiff[1:]:
        #print(i)
        temp_ov = downs(l, sample_para_h,sample_para_w)
        ov = np.hstack((ov, temp_ov))
        
        
        
    oh = (oh - 32768.0) * 0.8 * 0.001
    return oh,ov

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


# threshold 为只显示多少微米一下的器件间距，单位为微米
def show_component_distance(component_distance,sample_para_w,sample_para_h,img,threshold):
    indx=component_distance
    #  针对每一条轮廓,向上下 正负3个点，一共7个点计算方差，如果方差满足小于某个值的条件，则认为该点处于某个器件的相对中间位置,对其画间隙
    #  这样可以过滤掉列上面的很多杂乱数据
    
    #  采样频率为10的时候 相当于向上搜索 5微米*3 = 15微米  意味着要画出的器件的最小长度为15*2=30(上下两边) 微米，太小的就不画了，可能是噪音
    #w=3   # 3 是针对采样频率10调出来的  如果 采样频率为5  那么  w 乘以 (10/5) 倍
    
    w=3
    
    w=w*int(10/sample_para_h)
    left_border=indx[:,1]
    right_border=indx[:,2]
    
    var_l=[]
    for i in range(len(left_border)):
        if i>=w and i<len(left_border)-(w+1):
            ab=left_border[i-w:i+w+1]
            #print(ab)
            var_l.append(np.var(ab))
        else:
            var_l.append(0)
            
    var_r=[]
    for i in range(len(right_border)):
        if i>=w and i<len(right_border)-(w+1):
            ab=right_border[i-w:i+w+1]
            #print(ab)
            var_r.append(np.var(ab))
        else:
            var_r.append(0)
                
            
            
    Search=pd.DataFrame()
    Search['i']=indx[:, 0]
    Search['l']=indx[:, 1]
    Search['r']=indx[:, 2]
    Search['d']=indx[:, 3]
    Search['var_l']=var_l
    Search['var_r']=var_r
    

    choosed_lunkuo=[]
    #indx = indx[[not np.all(indx[i] == 0) for i in range(indx.shape[0])], :]
    #indx = np.array(sorted(indx, key=lambda x: x[3]))
    for j in range(len(Search)):
        if Search['var_l'][j]<10 and Search['var_r'][j]<10 and (Search['d'][j])*sample_para_w*0.5<threshold: # 上下方差小于20 且 最小距离小于thredhold微米才画出来
            choosed_lunkuo.append(j)
    distbox = []
    for i in range(len(choosed_lunkuo)-1):
        j=choosed_lunkuo[i]
        if choosed_lunkuo[i+1]-choosed_lunkuo[i]>1:

            real_distance=indx[j,3]*sample_para_w*0.5 # 降采样频率为 计算成真实的微米
            if real_distance > 0:
                distbox = np.append(distbox, real_distance)  # 间隙保存到distbox排序
                #cv2.putText(img, str(indx[j,3])+'微米',[int(np.mean(indx[j, 1:2])), int(indx[j, 0])], cv2.FONT_HERSHEY_COMPLEX, 1, [255,255,0])
                img = cv2ImgAddText(img, str(real_distance)+'微米', int(np.mean(indx[j, 1:2])), int(indx[j, 0]), (255, 255, 0), 20)
                pt1 = [int(indx[j, 1]), int(indx[j, 0])]
                pt2 = [int(indx[j, 2]), int(indx[j, 0])]
                cv2.arrowedLine(img, pt1, pt2, (0, 0, 255), 2)
    cv2.imwrite("findcomponent.jpg", img)
    if len(distbox)>0:
        print("器件最小间隙：" + str(np.min(distbox)))
    return img,Search


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


# 翻转180度
def flip180(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr



# 输入 color_rgb  输出板子的中心位置
def cal_chip_center(color_rgb):
    image = cv2.imread(color_rgb)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # 找到边界坐标
        x, y, w, h = cv2.boundingRect(c)  # 计算点集最外面的矩形边界
        if w>1000 and h>1000:
            #print(x, y, w, h)
        # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # cv2.imwrite("img_1.jpg", image)
        # 找面积最小的矩形
            rect = cv2.minAreaRect(c)
            # 得到最小矩形的坐标
            box = cv2.boxPoints(rect)
            # 标准化坐标到整数
            box = np.int0(box)
            return [0.5*(min(box[:,0])+max(box[:,0])),0.5*(min(box[:,1])+max(box[:,1]))]

            #return [box[0,0],box[0,1]]




    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#




def main(tiff_height_address,tiff_luminance_address,sample_para_h,sample_para_w):
    oh,ov=I=Tiff2_oh_ov(tiff_height_address, tiff_luminance_address,sample_para_h,sample_para_w)

    '''
    for i in range(oh.shape[0]):
        for j in range(oh.shape[1]):
            if oh[i][j]<-10:  # 高度值小于-10的无效数据填补
                oh[i][j]=oh[i][j-1]
            if ov[i][j]==0:
                ov[i][j]=ov[i][j-1]
    
    
    
    
    
    for i in range(oh.shape[0]):
        
        
        for j in range(oh.shape[1]):
            if oh[i][j]<-10:  # 高度值小于-10的无效数据填补
                oh[i][j]=1
            else:
                oh[i][j]=0
                
                
            if ov[i][j]==0:
                ov[i][j]=1
            else:
                ov[i][j]=0
                
    '''
    #color_rgb=numpy2rgb(oh,ov)
    #cv2.imwrite("color_rgb_z.jpg", color_rgb)
    return oh,ov

#############################################################
#####################start###################################
#############################################################

# 是否用180度的数据对其
open_duiqi=0


sample_para_h=10

sample_para_w=10

time_start=time.time()



path_height='D:/find1026/find1026/tiff_height/'
path_luminance='D:/find1026/find1026/tiff_luminance/'


oh,ov=main(path_height,path_luminance,sample_para_h,sample_para_w)


if open_duiqi==1:
    #oh,ov=main('D:/find1026/20221026/tiffheight/','D:/find1026/20221026/tiffluminance/',sample_para_h,sample_para_w)
    color_rgb=numpy2rgb(oh,ov)
    cv2.imwrite("color_rgb_z.jpg", color_rgb)

    #img = cv2.imread('color_rgb_f.jpg')
    oh_f,ov_f=main(path_height,path_luminance,sample_para_h,sample_para_w)


    oh_f=flip180(oh_f)
    ov_f=flip180(ov_f)

    color_rgb=numpy2rgb(oh_f,ov_f)
    cv2.imwrite("color_rgb_f.jpg", color_rgb)

    [center_x_z,center_y_z]=cal_chip_center('color_rgb_z.jpg')
    [center_x_f,center_y_f]=cal_chip_center('color_rgb_f.jpg')

    [delta_x,delta_y]=[int(center_x_f-center_x_z),int(center_y_f-center_y_z)]

    oh_completion=oh
    ov_completion=ov


    for i in range(oh_completion.shape[0]):
        for j in range(oh_completion.shape[1]):
            if oh_completion[i][j]<-10:  # 高度值小于-10的无效数据填补
                oh_completion[i][j]=oh_f[i+delta_x][j+delta_y]
            if ov_completion[i][j]==0:
                ov_completion[i][j]=ov_f[i+delta_x][j+delta_y]
    color_rgb=numpy2rgb(oh_completion,ov_completion)
    cv2.imwrite("color_rgb_completion.jpg", color_rgb)


else:
    color_rgb=numpy2rgb(oh,ov)
    cv2.imwrite("color_rgb.jpg", color_rgb)
    component_distance = componemt_dist(oh,sample_para_w)
    img = cv2.imread('color_rgb.jpg')
    img,Search_Component=show_component_distance(component_distance,sample_para_w,sample_para_h,img,500)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


def show_pins_distance(pins_distance1, img):
    # cv2.imwrite("2.jpg", img)
    left_border = pins_distance1[:, 1]
    right_border = pins_distance1[:, 2]
    w = 1
    var1 = []
    var2 = []
    for i in range(len(left_border)):
        if i >= w and i < len(left_border) - w:
            ab = left_border[i - w:i + w + 1]
            cd = right_border[i - w:i + w + 1]

            # print(ab)
            var1.append(np.var(ab))
            var2.append(np.var(cd))
        else:
            var1.append(0)
            var2.append(0)

    Pins_distance1 = pd.DataFrame()
    Pins_distance1['row'] = pins_distance1[:, 0]
    Pins_distance1['left'] = pins_distance1[:, 1]
    Pins_distance1['right'] = pins_distance1[:, 2]
    Pins_distance1['distance'] = pins_distance1[:, 3]
    Pins_distance1['n'] = pins_distance1[:, 4]
    Pins_distance1['var1'] = var1
    Pins_distance1['var2'] = var2

    choosed_lunkuo = []

    # indx = indx[[not np.all(indx[i] == 0) for i in range(indx.shape[0])], :]
    # indx = np.array(sorted(indx, key=lambda x: x[3]))
    j = 0
    distbox = []
    while (j < len(Pins_distance1)):

        if Pins_distance1['var1'][j] < 50 and Pins_distance1['var2'][j] < 50 and Pins_distance1['n'][
            j] > 2:  # 上下方差小于1 且 最小距离小于20才画出来
            # if 1:
            # choosed_lunkuo.append(Pins_distance1['i'][j])
            # print(int(Pins_distance1['left'][j]),int(Pins_distance1['row'][j]))
            pt1 = [int(Pins_distance1['left'][j]), int(Pins_distance1['row'][j])]
            pt2 = [int(Pins_distance1['right'][j]), int(Pins_distance1['row'][j])]

            real_distance = Pins_distance1['distance'][j] * 10 * 0.5  # 降采样频率为10 计算成真实的微米
            img = cv2ImgAddText(img, str(real_distance) + '微米', int(Pins_distance1['left'][j]),
                                int(Pins_distance1['row'][j]) + 10, (255, 255, 0), 20)
            distbox = np.append(distbox, real_distance)
            cv2.arrowedLine(img, pt1, pt2, (0, 255, 0), 1)
            # cv2.imwrite("./datasets/resultImg/findpins.jpg", img)
            j = j + 20  # 5个轮廓之间只显示一个
        j = j + 1
    cv2.imwrite("./datasets/resultImg/findpins.jpg", img)
    if len(distbox) > 0:
        print("最小引脚平均间隙：" + str(np.min(distbox)))
    else:
        print("未发现引脚")
    return Pins_distance1

# img = cv2.imread('color_rgb.jpg')


pins_distance1 = find_pins.pins_dist(oh)
b=show_pins_distance(pins_distance1, img)

