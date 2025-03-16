import numpy as np
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as KNN

def img_to_vector(filename):
    returnVect=np.zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])

    return returnVect

def handwritingClassTest():
    hwLabels=list()
    trainingFileList=listdir('trainingDigits')
    m=len(trainingFileList)
    trainingMat=np.zeros((m,1024))

    for i in range(m):
        fileNameStr=trainingFileList[i]
        classNumber=int(fileNameStr.split('_')[0])
        hwLabels.append(classNumber)
        trainingMat[i,:]=img_to_vector('trainingDigits/%s' % fileNameStr)

    neigh=KNN(n_neighbors=5,algorithm='auto')
    neigh.fit(trainingMat,hwLabels)
    testFileList=listdir('testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        classNumber=int(fileNameStr.split('_')[0])
        vectorUnderTest=img_to_vector('testDigits/%s' % fileNameStr)
        classifierResult=neigh.predict(vectorUnderTest)
        print(f"the classifier came back with:{classifierResult},the real answer is:{classNumber}")
        if classifierResult!=classNumber:
            errorCount+=1
    print(f"the total number of errors is:{errorCount}")
    print(f"the total error rate is:{errorCount/mTest*100}")

if __name__=="__main__":
    handwritingClassTest()
