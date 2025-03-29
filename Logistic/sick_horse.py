import numpy as np
import os
import matplotlib.pyplot as plt

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

def sigmoid(inX):
    inX=np.clip(inX,-50,50)
    return 1.0/(1+np.exp(-inX))

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

def gradAscent(dataMatIn,classLabels):
    dataMatrix=np.asmatrix(dataMatIn)
    labelMat=np.asmatrix(classLabels).transpose()
    m,n=np.shape(dataMatrix)
    alpha=0.001
    maxCycles=500
    weights=np.ones((n,1))
    weights_array=np.array([])
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)
        error=(labelMat-h)
        weights=weights+alpha*dataMatrix.transpose()*error
        weights_array=np.append(weights_array,weights)
    weights_array=weights_array.reshape(maxCycles,n)
    return weights.getA(),weights_array

def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n=np.shape(dataMatrix)
    weights=np.ones(n)
    weights_array=np.array([])
    for j in range(numIter):
        dataIndex=list(range(m))
        for i in range(m):
            alpha=4/(1.0+j+i)+0.01
            randIndex=int(np.random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(dataMatrix[randIndex]*weights))
            error=classLabels[randIndex]-h
            weights=weights+alpha*error*dataMatrix[randIndex]
            weights_array=np.append(weights_array,weights,axis=0)
            del dataIndex[randIndex]

        
    weights_array=weights_array.reshape(numIter*m,n)
    return weights,weights_array

def plotWeights(weights_array1,weights_array2):
    fig,axs=plt.subplots(nrows=3,ncols=2,sharex=False,sharey=False,figsize=(20,10))
    x1=np.arange(0,len(weights_array1),1)
    axs[0][0].plot(x1,weights_array1[:,0])
    axs0_title_text=axs[0][0].set_title('Gradient ascent algorithm:relation of weights and interations')
    axs0_ylabel_text=axs[0][0].set_ylabel('W0')
    plt.setp(axs0_title_text,size=20,weight='bold',color='black')
    plt.setp(axs0_ylabel_text,size=20,weight='bold',color='black')


    axs[1][0].plot(x1,weights_array1[:,1])
    axs1_ylabel_text=axs[1][0].set_ylabel('W1')
    plt.setp(axs1_ylabel_text,size=20,weight='bold',color='black')


    axs[2][0].plot(x1,weights_array1[:,2])
    axs2_xlabel_text = axs[2][0].set_xlabel(u'num of iterations')
    axs2_ylabel_text = axs[2][0].set_ylabel(u'W1')
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black') 
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')


    x2 = np.arange(0, len(weights_array2), 1)
    #绘制w0与迭代次数的关系
    axs[0][1].plot(x2,weights_array2[:,0])
    axs0_title_text = axs[0][1].set_title(u'advanced gradient ascent')
    axs0_ylabel_text = axs[0][1].set_ylabel(u'W0')
    plt.setp(axs0_title_text, size=20, weight='bold', color='black') 
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    #绘制w1与迭代次数的关系
    axs[1][1].plot(x2,weights_array2[:,1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'W1')
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    #绘制w2与迭代次数的关系
    axs[2][1].plot(x2,weights_array2[:,2])
    axs2_xlabel_text = axs[2][1].set_xlabel(u'num of iterations')
    axs2_ylabel_text = axs[2][1].set_ylabel(u'W1')
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black') 
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    plt.show()       

def classifyVector(inX,weights):
    product=sum(inX*weights)
    prob=sigmoid(sum(inX*weights))
    if prob>0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    current_file=os.path.dirname(__file__)
    train_file_path=os.path.join(current_file,'horseColicTraining.txt')
    test_file_path=os.path.join(current_file,'horseColicTest.txt')
    trainingSet=[]
    trainingLabels=[]
    train_file=open(train_file_path,'r')
    test_file=open(test_file_path,'r')
    
    for line in train_file.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    
    trainWeights,train_weights_array=stocGradAscent1(np.array(trainingSet),trainingLabels,150)

    errorCount=0
    numTestVec=0.0

    for line in test_file.readlines():
        numTestVec+=1.0
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr),trainWeights))!=int(currLine[-1]):
            errorCount+=1
    errorRate=(float(errorCount)/numTestVec)*100
    print("the error rate of this test is:%f%%"%errorRate)
    return errorRate

def colicTest_origin():
    current_file=os.path.dirname(__file__)
    train_file_path=os.path.join(current_file,'horseColicTraining.txt')
    test_file_path=os.path.join(current_file,'horseColicTest.txt')
    trainingSet=[]
    trainingLabels=[]
    train_file=open(train_file_path,'r')
    test_file=open(test_file_path,'r')
    
    for line in train_file.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    
    trainWeights,train_weights_array=gradAscent(np.array(trainingSet),trainingLabels)

    errorCount=0
    numTestVec=0.0

    for line in test_file.readlines():
        numTestVec+=1.0
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr),trainWeights[:,0]))!=int(currLine[-1]):
            errorCount+=1
    errorRate=(float(errorCount)/numTestVec)*100
    print("the error rate of this origin test is:%f%%"%errorRate)
    return errorRate


        
if __name__=='__main__':
    colicTest()
    colicTest_origin()
