# -*- coding:utf-8 -*-
# __author__ = 'CaoRui'
import numpy as np
#创建数据集
def loadDataSet():
    postingList = [['my','dog','has','flea','problem','help','please'],\
                   ['maybe','not','take','him','to','dog','park','stupid'],\
                   ['my','dalmation','is','so','cute','I','love','him'],\
                   ['stop','posting','stupid','worthless','garbage'],\
                   ['mr','licks','ate','my','steak','how','to','stop','hom'],\
                   ['quit','buying','worthless','dog','food','stupid']]
    classVec=[0,1,0,1,0,1]
    return postingList, classVec

#数据去重
def createVocabList(dataSet):
    vocabList = set([])
    for example in dataSet:
        vocabList = vocabList | set(example)
    return list(vocabList)

#词条向量化
def setOfWords2Vec(vocabList, InputSet):
    returnVec = [0] * len(vocabList)
    for word in InputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    return returnVec

#从词向量计算概率
def trainNBO(trainMatrix, trainCategory):
    numTrainDoc = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDoc)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in xrange(numTrainDoc):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive

#贝叶斯分类
def classifyNB(vect2Classify, p0Vect,p1Vect,pClass):
    p1 = sum(vect2Classify * p1Vect) + np.log(pClass)
    p0 = sum(vect2Classify * p0Vect) + np.log(1-pClass)
    if p1 > p0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    listOpost,listClass = loadDataSet()
    myVocabList = createVocabList(listOpost)
    for word in myVocabList:
        print word
    trainMat = []
    for document in listOpost:
        trainMat.append(setOfWords2Vec(myVocabList, document))
    p0Vect, p1Vect, pAbusive = trainNBO(np.array(trainMat), np.array(listClass))
    testEntry = ['love','my','dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classfied as: ',classifyNB(thisDoc,p0Vect,p1Vect,pAbusive)
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classfied as: ', classifyNB(thisDoc, p0Vect, p1Vect, pAbusive)