import matplotlib.pyplot as plt
from math import log
import pickle
import operator

def calcSHannonEnt(dataset):
    numEntires=len(dataset)
    LabelCounts=dict()
    for featVec in dataset:
        currentLabel=featVec[-1]
        if currentLabel not in LabelCounts.keys():
            LabelCounts[currentLabel]=0

        LabelCounts[currentLabel]+=1

    shannonEnt=0.0
    for key in LabelCounts:
        prob=float(LabelCounts[key])/numEntires
        shannonEnt-=prob*log(prob,2)
    return shannonEnt

def createDataSet():
	dataSet = [[0, 0, 0, 0, 'no'],						#数据集
			[0, 0, 0, 1, 'no'],
			[0, 1, 0, 1, 'yes'],
			[0, 1, 1, 0, 'yes'],
			[0, 0, 0, 0, 'no'],
			[1, 0, 0, 0, 'no'],
			[1, 0, 0, 1, 'no'],
			[1, 1, 1, 1, 'yes'],
			[1, 0, 1, 2, 'yes'],
			[1, 0, 1, 2, 'yes'],
			[2, 0, 1, 2, 'yes'],
			[2, 0, 1, 1, 'yes'],
			[2, 1, 0, 1, 'yes'],
			[2, 1, 0, 2, 'yes'],
			[2, 0, 0, 0, 'no']]
	labels = ['Age', 'Has Job', 'Has House', 'Credit']		#特征标签（改为英文）
	return dataSet, labels 	


def splitDataSet(dataset,axis,value):
     retDataSet=list()
     for featVec in dataset:
          if featVec[axis]==value:
               reducedFeatVec=featVec[:axis]
               reducedFeatVec.extend(featVec[axis+1:])
               retDataSet.append(reducedFeatVec)

     return retDataSet

def chooseBestFeatureToSplit(dataset):
     numFeatures=len(dataset[0])-1
     baseEntropy=calcSHannonEnt(dataset=dataset)
     bestInfoGain=0
     bestFeature=-1

     for i in range(numFeatures):
          featList=[example[i] for example in dataset]
          uniqueVals=set(featList)
          newEntropy=0

          for value in uniqueVals:
               subDataSet=splitDataSet(dataset=dataset,axis=i,value=value)
               prob=len(subDataSet)/float(len(dataset))
               newEntropy+=prob*calcSHannonEnt(dataset=subDataSet)

          infoGain=baseEntropy-newEntropy
          print(f"The infoGain of labels {[i]} is {infoGain}")
          if infoGain>bestInfoGain:
               bestInfoGain=infoGain
               bestFeature=i
               
     return bestFeature


def majorityCnt(classList):
    classCount=dict()
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1

    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


def createTree(dataset,labels,featLabels):
    classList=[example[-1] for example in dataset]
    if classList.count(classList[0])==len(classList):
          return classList[0]
     
    if len(dataset[0])==1 or len(labels)==0:
          return majorityCnt(classList=classList)
     
    bestFeat=chooseBestFeatureToSplit(dataset=dataset)
    bestFeatLabel=labels[bestFeat]
    featLabels.append(bestFeatLabel)
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataset]
    uniqueVal=set(featValues)
    for value in uniqueVal:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataset=dataset,axis=bestFeat,value=value),subLabels,featLabels)
    return myTree
     
def getNumLeafs(myTree):
    numLeafs=0
    firstStr=next(iter(myTree))
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs+=getNumLeafs(secondDict[key])
        else:
             numLeafs+=1

    return numLeafs
        
def getTreeDepth(myTree):
    maxDepth=0
    firstStr=next(iter(myTree))
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth=1+getTreeDepth(secondDict[key])
        else:
            thisDepth=1
        if thisDepth>maxDepth:
            maxDepth=thisDepth
    return maxDepth

def createPlot(inTree):
     fig=plt.figure(1,facecolor='white')
     fig.clf()
     axprops=dict(xticks=[],yticks=[])
     createPlot.ax1=plt.subplot(111,frameon=False,**axprops)
     plotTree.totalW=float(getNumLeafs(inTree))
     plotTree.totalD=float(getTreeDepth(inTree))
     plotTree.xOff=-0.5/plotTree.totalW
     plotTree.yOff=1.0
     plotTree(inTree,(0.5,1.0),'')
     plt.show()

def plotMidText(cntrPt, parentPt, txtString):
	xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]											#计算标注位置					
	yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
	createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)
     
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
	arrow_args = dict(arrowstyle="<-")											#定义箭头格式
	createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',	#绘制结点
		xytext=centerPt, textcoords='axes fraction',
		va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, fontsize=10)

def plotTree(myTree,parentPt,nodeTxt):
    decisionNode=dict(boxstyle="sawtooth",fc="0.8")
    leafNode=dict(boxstyle="round4",fc="0.8")
    numLeaf=getNumLeafs(myTree=myTree)
    depth=getTreeDepth(myTree=myTree)
    firstStr=next(iter(myTree))
    cntrPt=(plotTree.xOff+(1.0+float(numLeaf))/2.0/plotTree.totalW,plotTree.yOff)
    plotMidText(cntrPt=cntrPt,parentPt=parentPt,txtString=nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict=myTree[firstStr]
    plotTree.yOff=plotTree.yOff-1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff=plotTree.xOff+1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    plotTree.yOff=plotTree.yOff+1.0/plotTree.totalD


def classify(inputTree,featLabels,testVec):
    firstStr=next(iter(inputTree))
    secondDict=inputTree[firstStr]
    featIndex=featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(inputTree=secondDict[key],featLabels=featLabels,testVec=testVec)
            else:
                classLabel=secondDict[key]
    return classLabel


def storeTree(inputTree,filename):
     with open(filename,'wb') as fw:
          pickle.dump(inputTree,fw)

def grabTree(filename):
     fr=open(filename,'rb')
     return pickle.load(fr)


if __name__=="__main__":
    dataset,labels=createDataSet()
    # 保存原始特征标签的副本，用于后续分类
    labels_copy = labels.copy()
    featLabels=[]
    myTree=createTree(dataset=dataset,labels=labels.copy(),featLabels=featLabels)
    # createPlot(inTree=myTree)
    restVec=[0,1,1,2]
    # 使用原始特征标签进行分类
    result=classify(inputTree=myTree,featLabels=labels_copy,testVec=restVec)
    if result=='yes':
        print('Approved')
    if result=='no':
        print('Rejected')
    createPlot(inTree=myTree)
    #  print(chooseBestFeatureToSplit(dataset=dataset))
    #  print(calcSHannonEnt(dataset=dataset))

