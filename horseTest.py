#coding:utf-8
__author__ = 'FireJohnny'

from numpy import *


def stocGradAscent(dataMatrix,classLabel,numIter):
    m, n = shape(dataMatrix)
    weights = ones(n)

    for j  in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0 + j+ i) +0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error  = classLabel[randIndex] - h
            weights = weights + alpha * error *dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights



def sigmoid(inX):
    return 1 / (1 + exp( -inX ))


def classifyVector(inX,weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest(fileName_1,fileName_2):

    trainingSet = []
    trainingLabel = []
    with open(fileName_1) as frTrain:
        for line in frTrain.readlines():
            currLine = line.strip().split('\t')
            lineList = []
            for i in range(21):
                lineList.append(float(currLine[i]))
            trainingSet.append(lineList)
            trainingLabel.append(float(currLine[21]))
    trainWeights = stocGradAscent(array(trainingSet),trainingLabel,500)
    errorCount = 0
    numTestVec = 0.0
    with open(fileName_2) as frTest:
        for line in frTest.readlines():
            numTestVec += 1.0
            currLine = line.strip().split('\t')
            lineList = []
            for i in range(21):
                lineList.append(float(currLine[i]))

            if int(classifyVector(array(lineList), trainWeights)) != int(currLine[21]):
                errorCount +=1
    errorRate = (float(errorCount)/numTestVec)
    print ("the error rate of this test is :%s") %errorRate
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest("horseColicTraining.txt","horseColicTest.txt")
    print "after %d iterations the averge error rate is : %f"%(numTests,errorSum/float(numTests))

if __name__ == "__main__":
    multiTest()