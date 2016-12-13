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
        error = LabelMat - h
        weights = weights + alpha * dataMat.transpose() * error

    return np.array(weights)

def stoGradAscent0(dataArr,labelArr):
    dataMat = np.array(dataArr)
    #labelMat  = np.mat(labelArr).transpose()
    m, n =  np.shape(dataMat)
    alpha = 0.001
    weights = np.ones(n)

    for i in range(m):
        h = sigmoid(sum(dataMat * weights))
        error = labelArr[i] - h
        weights = weights + alpha * error * dataMat[i]

    return np.array(weights)

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    #print type(dataMatrix)
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01    #apha decreases with iteration, does not
            randIndex = int(np.random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex]) #计算完的样本就进行删除就好
    return weights

def poltBestFit():

    dataArr , labelArr = loadFile("testSet.txt")
    dataArr = np.array(dataArr)
    #weights = gradscent(dataArr,labelArr)
    #weights = gradscent(dataArr,labelArr)

    weights = stocGradAscent1(dataArr,labelArr)
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




'''
def colicTest():
    frTrain = open("horseColicTraining.txt")
    frTest = open("horseColicTest.txt")
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))

        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stoGradAscent0(np.array(trainingSet),trainingLabels,500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr),trainWeights)) != int(currLine[21]):
            errorCount +=1
    errorRate = (float(errorCount)/numTestVec)
    print errorRate
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
        print "%d   ...   %f"(numTests,errorSum/(float(numTests)))
'''


if __name__ == "__main__":

    pol = poltBestFit()