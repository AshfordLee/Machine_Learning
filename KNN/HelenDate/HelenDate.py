import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib
import operator
import os
import pandas as pd
matplotlib.use('TkAgg')


def file_to_matrix(filename):
    fr=open(file=filename,mode='r',encoding='utf-8')
    arrayOLines=fr.readlines()
    arrayOLines[0]=arrayOLines[0].lstrip('\ufeff')
    numberOfLines=len(arrayOLines)
    returnMat=np.zeros((numberOfLines,3))
    classLabelVector=list()
    index=0


    for line in arrayOLines:
        line=line.strip()
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        if listFromLine[-1]=="didntLike":
            classLabelVector.append(1)
        
        elif listFromLine[-1]=="smallDoses":
            classLabelVector.append(2)
        
        elif listFromLine[-1]=="largeDoses":
            classLabelVector.append(3)
        
        index+=1

    return returnMat,classLabelVector


def showdatas(datingDataMat,datingLabels):
    # 设置字体
    matplotlib.rcParams['axes.unicode_minus'] = False

    fig,axs=plt.subplots(nrows=2,ncols=2,sharex=False,sharey=False,
                         figsize=(13,8))
    

    numberOfLabels=len(datingLabels)
    LabelsColors=[]
    for i in datingLabels:
        if i==1:
            LabelsColors.append('black')
        
        elif i==2:
            LabelsColors.append('orange')

        elif i==3:
            LabelsColors.append('red')

        
    axs[0][0].scatter(x=datingDataMat[:,0],y=datingDataMat[:,1],color=LabelsColors,s=15,alpha=0.5)
    axs0_title_text=axs[0][0].set_title('Flight Miles vs. Video Game Time')
    axs0_xlabel_text=axs[0][0].set_xlabel('Flight Miles Per Year')
    axs0_ylabel_text=axs[0][0].set_ylabel('Video Game Time Percentage')

    plt.setp(axs0_title_text,size=9,weight='bold',color='red')
    plt.setp(axs0_xlabel_text,size=7,weight='bold',color='black')
    plt.setp(axs0_ylabel_text,size=7,weight='bold',color='black')

    axs[0][1].scatter(x=datingDataMat[:,0],y=datingDataMat[:,2],color=LabelsColors,s=15,alpha=0.5)
    axs1_title_text=axs[0][1].set_title('Flight Miles vs. Ice Cream Consumption')
    axs1_xlabel_text=axs[0][1].set_xlabel('Flight Miles Per Year')
    axs1_ylabel_text=axs[0][1].set_ylabel('Liters of Ice Cream Per Week')

    plt.setp(axs1_title_text,size=9,weight='bold',color='red')
    plt.setp(axs1_xlabel_text,size=7,weight='bold',color='black')
    plt.setp(axs1_ylabel_text,size=7,weight='bold',color='black')

    axs[1][0].scatter(x=datingDataMat[:,1],y=datingDataMat[:,2],color=LabelsColors,s=15,alpha=0.5)
    axs2_title_text=axs[1][0].set_title('Video Game Time vs. Ice Cream Consumption')
    axs2_xlabel_text=axs[1][0].set_xlabel('Video Game Time Percentage')
    axs2_ylabel_text=axs[1][0].set_ylabel('Liters of Ice Cream Per Week')

    plt.setp(axs2_title_text,size=9,weight='bold',color='red')
    plt.setp(axs2_xlabel_text,size=7,weight='bold',color='black')
    plt.setp(axs2_ylabel_text,size=7,weight='bold',color='black')

    didntLike=mlines.Line2D([], [], color='black', marker='.',markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',markersize=6, label='largeDoses')
    axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
    plt.show()
    
    
def autonorm(dataSet:pd.DataFrame):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=np.zeros(np.shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-np.tile(minVals,(m,1))
    normDataSet=normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def classify0(inX,dataset,labels,k):
    dataSetSize=dataset.shape[0]
    diffMat=np.tile(inX,(dataSetSize,1))-dataset
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndices=distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndices[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


def datingClassTest():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建数据文件的完整路径
    data_file = os.path.join(script_dir, 'datingTestSet.txt')
    datingDataMat,datingLabels=file_to_matrix(data_file)
    hoRatio=0.1
    norMat,ranges,minVals=autonorm(datingDataMat)
    m=norMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0

    for i in range(numTestVecs):
        classifierResult=classify0(norMat[i,:],norMat[numTestVecs:m,:],datingLabels[numTestVecs:m],4)
        print(f"the classifier came back with:{classifierResult},the real answer is:{datingLabels[i]}")
        if classifierResult!=datingLabels[i]:
            errorCount+=1
    print(f"the total error rate is:{errorCount/float(numTestVecs)*100}")


def classifyPerson():
    resultList=['dislike','smallDoses','LargeDoses']
    precentats=float(input("precentats of time spent playing video games?"))
    ffMiles=float(input("frequent flier miles earned per year?"))
    iceCream=float(input("liters of ice cream consumed per week?"))
    datingDataMat,datingLabels=file_to_matrix(data_file)
    norMat,ranges,minVals=autonorm(datingDataMat)
    inArr=np.array([ffMiles,precentats,iceCream])
    classifierResult=classify0((inArr-minVals)/ranges,norMat,datingLabels,3)
    print(f"You will probably like this person:{resultList[classifierResult-1]}")
    
if __name__=="__main__":
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建数据文件的完整路径
    data_file = os.path.join(script_dir, 'datingTestSet.txt')
    # 使用完整路径
    datingDataMat,datingLabels=file_to_matrix(data_file)
    afternorm=autonorm(dataSet=datingDataMat)
    datingClassTest()
    # print(afternorm)
    # showdatas(datingDataMat,datingLabels)