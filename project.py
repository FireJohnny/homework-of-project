#coding:utf-8
__author__ = 'FireJohnny'

import numpy as np
from matplotlib import *

def loadFile(fileName):
    dataArr = []
    LabelArr = []
    with open(fileName) as fr:
        for line in fr.readlines():
            lineArr = line.strip().split()
            #print lineArr
            dataArr.append([1.0,float(lineArr[0]),float(lineArr[1])])
            LabelArr.append(int(lineArr[2]))

    return dataArr,LabelArr

dataArr,LabelArr = loadFile("testSet.txt")
print dataArr,"\n",LabelArr

    