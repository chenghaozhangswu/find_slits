# -*- coding: utf-8 -*-
"""
@Time ： 2022/11/12 15:34
@Auth ： zch171041@outlook.com
@File ：get_pcloud.py
@IDE ：PyCharm
@python version:3.6
@Motto：ABC(Always Be Coding)
"""
import pandas as pd
import os

# x1,y1表示传入起始点坐标，x2，y2表示传入终点坐标
# Name1表示扫描的整张pcb板的点云excel带格式后缀全名
# Name2表示从扫描的整张pcb板点云数据中截取指定区域数据后要保存为excel的名字
def getPoints(x1,y1,x2,y2,Name1,Name2):
    data = pd.read_csv("./datasets/csv/{}".format(Name1)).values
    data_new = data[x1:x2,y1:y2]
    path = './datasets/excel/'
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
    data_new = pd.DataFrame(data_new)
    data_new.to_excel(excel_writer=r"./datasets/excel/{}".format(Name2), header=False, index=False)


getPoints(100,100,101,101,'Name.csv','oh_new.xlsx')
