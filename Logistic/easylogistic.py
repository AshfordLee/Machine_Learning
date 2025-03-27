import matplotlib.pyplot as plt
import numpy as np
import os


def loadDataSet():
    dataMat=[]
    labelMat=[]
    current_file=os.path.dirname(__file__)
    file_path=os.path.join(current_file,'testSet.txt')
    with open(file_path,'r') as f:
        for line in f.readlines():
            lineArr=line.strip().split()
            dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
    f.close()
    return dataMat,labelMat

def plotDataSet():
    dataMat,labelMat=loadDataSet()
    dataArr=np.array(dataMat)
    n=np.shape(dataArr)[0]
    xcord1=[]
    ycord1=[]
    xcord2=[]
    ycord2=[]

    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s',alpha=0.5)
    ax.scatter(xcord2,ycord2,s=30,c='green',alpha=0.5)
    plt.title('DataSet')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMatIn,classLabels):
    dataMatrix=np.asmatrix(dataMatIn)
    labelMat=np.asmatrix(classLabels).transpose()
    m,n=np.shape(dataMatrix)
    alpha=0.001
    maxCycles=500
    weights=np.ones((n,1))
    for k in range(maxCycles):
        product=dataMatrix*weights
        h=sigmoid(product)
        error=(labelMat-h)
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights.getA()

    
def plotBestFit(weights):
    dataMat,labelMat=loadDataSet()
    dataArr=np.array(dataMat)
    n=np.shape(dataArr)[0]
    xcord1=[]
    ycord1=[]
    xcord2=[]
    ycord2=[]   
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s',alpha=0.5)
    ax.scatter(xcord2,ycord2,s=30,c='green',alpha=0.5)
    x=np.arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.title('BestFit')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    
if __name__=='__main__':
    dataMat,labelMat=loadDataSet()
    weights=gradAscent(dataMat,labelMat)
    plotBestFit(weights)