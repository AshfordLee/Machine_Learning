import numpy as np
import pandas as pd
from functools import reduce

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],                #切分的词条
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]                                                                   #类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList,classVec


def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet | set(document)
    return list(vocabSet)

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    p0Num=np.ones(numWords)
    p1Num=np.ones(numWords)
    p0Denom=2.0
    p1Denom=2.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])

        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])

    p1Vect=np.log(p1Num/p1Denom)
    p0Vect=np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)       
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def testingNB():
    postingList,classVec=loadDataSet()
    print("PostingList:\n",postingList)
    myVocabList=createVocabList(postingList)
    print("myVocabList:\n",myVocabList)
    trainMat=[]
    for postionDoc in postingList:
        trainMat.append(setOfWords2Vec(myVocabList,postionDoc))

    p0V,p1V,pAb=trainNB0(trainMat,classVec)
    print("p0V:\n",p0V)
    print("p1V:\n",p1V)
    print("pAb:\n",pAb)
    testEntry=['Love','my','dalmation']
    thisDoc=np.array(setOfWords2Vec(myVocabList,testEntry))
    if classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry,"属于侮辱类")
    else:
        print(testEntry,"属于非侮辱类")

    testEntry=['stupid','garbage']
    thisDoc=np.array(setOfWords2Vec(myVocabList,testEntry))
    if classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry,"属于侮辱类")
    else:
        print(testEntry,"属于非侮辱类")

if __name__=="__main__":
    testingNB()
    