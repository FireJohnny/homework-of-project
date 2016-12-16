#coding:utf-8
__author__ = 'FireJohnny'

import numpy as np
import matplotlib.pyplot as plt

def loadFile(fileName):
    dataArr = []
    LabelArr = []

    with open(fileName) as fr:
        for line in fr.readlines():
            lineArr = line.strip().split()
            #print lineArr
            dataArr.append([1.0,float(lineArr[0]), float(lineArr[1])])
            LabelArr.append(int(lineArr[2]))

    return dataArr, LabelArr

def sigmoid(X):
    return 1/(1+np.exp(-X))

def gradscent(dataArr,LabelArr):
    dataMat = np.mat(dataArr)
    LabelMat = np.mat(LabelArr).transpose()

    m,n = np.shape(dataMat)
    #alpha = 0.01 #步长的确定
    alpha  = 0.001
    Cycles  = 200    #循环次数
    weights  = np.ones((n,1))

    for i in range(Cycles):
        h = sigmoid(dataMat * weights)
        #print type(h)
        error = LabelMat - h
        weights = weights + alpha * dataMat.transpose() * error #dataMat.transpose()*error 是梯度
        print weights

    return np.array(weights)

def stoGradAscent0(dataArr,labelArr):
    dataMat = np.array(dataArr)
    #labelMat  = np.mat(labelArr).transpose()
    m, n =  np.shape(dataMat)
    alpha = 0.01
    weights = np.ones(n)

    for i in range(m):
        h = sigmoid(sum(dataMat[i] * weights))
        print h
        error = labelArr[i] - h
        weights = weights + alpha * error * dataMat[i]

    return np.array(weights)

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    #print type(dataMatrix)
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(np.random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex]) #计算完的样本就进行删除就好
    return weights


def poltBestFit():

    dataArr , labelArr = loadFile("testSet.txt")
    dataArr = np.array(dataArr)
    #weights = gradscent(dataArr,labelArr)
    #weights = stoGradAscent0(dataArr,labelArr)

    #weights = stocGradAscent1(dataArr,labelArr)
    weights  = conjjugate(dataArr,labelArr)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelArr[i]) == 1 :
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] -weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('Y2')
    plt.show()

def classifyVector(intX , weights):
    prob =sigmoid(sum(intX * weights))
    if prob >0.5:return 1.0
    else : return 0.0


if __name__ == "__main__":

    pol = poltBestFit()