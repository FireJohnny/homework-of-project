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




def poltBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat ,labelMat = loadFile()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1 :
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker ='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = np.arange(-3.0,3.0,0.1)
    y = (-weights[0] -weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('Y2')
    plt.show()