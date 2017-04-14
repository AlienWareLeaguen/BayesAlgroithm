# -*- coding:utf-8 -*-
# __author__ = 'CaoRui'
import numpy as np
import re
import jieba
import feedparser
import operator
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

#词条向量化-词集模型
def setOfWords2Vec(vocabList, InputSet):
    returnVec = [0] * len(vocabList)
    for word in InputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    return returnVec

#词条向量化-词袋模型
def bagOfWords2Vec(vocabList, InputSet):
    returnVec = [0] * len(vocabList)
    for word in InputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
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

#文本解析器
def textParse(bigString):
    regEx = re.compile('\\w*')
    listOfToken = regEx.split(bigString)
    return [tok.lower() for tok in bigString if len(tok) > 2]
#去除中文标点
def chineseTextParse(bigString):
    bigString = bigString.decode("utf8")
    string = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+".decode("utf8"), "".decode("utf8"), bigString)
    print string

#垃圾邮件过滤
def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in xrange(1,26):
        wordList = textParse(open('C:\Users\CaoRui\Desktop\数据集\EmailNO\%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('C:\Users\CaoRui\Desktop\数据集\EmailOK\%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    #随机构建训练集
    trainningSet = range(50)
    testSet = []
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainningSet)))
        testSet.append(trainningSet[randIndex])
        del(trainningSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainningSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0Vect, p1Vect, pAbusive = trainNBO(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(wordVector, p0Vect, p1Vect, pAbusive) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ',float(errorCount)/len(testSet)

#Rss分类器及高频词去除函数
def calcMostFreq(vocabList, fullText):
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortFreq[:30]
def localWords(feed1, feed0):
    docList =[]
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainningSet = range(2*minLen)
    testSet = []
    for i in range(20):
        randIndex = int(np.random.uniform(0, len(trainningSet)))
        testSet.append(trainningSet[randIndex])
        del(trainningSet[randIndex])
    trainMat = []
    trainClasses = []
    for decIndex in trainningSet:
        trainMat.append(bagOfWords2Vec(vocabList, docList[decIndex]))
        trainClasses.append(classList[decIndex])
    p0V,p1V,pSpam = trainNBO(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for decIndex in testSet:
        wordVect = bagOfWords2Vec(vocabList, docList[decIndex])
        if classifyNB(np.array(wordVect), p0V, p1V, pSpam) != classList[decIndex]:
            errorCount += 1
    print 'the error rate is :',float(errorCount)/len(testSet)
    return vocabList, p0V, p1V

#最具表征性的词汇显示函数
def getTopWords(ny, sf):
    vocabList, p0V, p1V = localWords(ny, sf)
    topNy = []
    topSf = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSf.append((vocabList[i], p1V[i]))
        if p1V[i] > -6.0:
            topNy.append((vocabList[i], p0V[i]))
    sortedSF = sorted(topSf, key=lambda  pair: pair[1], reverse=True)
    print "sfsfsfsfsfsfsfsfsfsfsfsf"
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNy, key=lambda  pair: pair[1], reverse=True)
    print "nfnfnfnfnfnfnfnfnfnfnfnfn"
    for item in sortedNY:
        print item[0]

if __name__ == '__main__':
    # temp = "想做/ 兼_职/学生_/ 的 、加,我有,惊,喜,哦"
    # string = chineseTextParse(temp)
    # seg_list = jieba.cut(string,cut_all=False)
    # print "Dafault Mode:","/".join(seg_list)
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    #vocabList, pSF, pNY = localWords(ny, sf)
    getTopWords(ny, sf)

