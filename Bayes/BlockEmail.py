import re
import numpy as np


def textParse(bigString):
    """
    根据垃圾邮件格式优化的文本解析函数
    1. 使用更复杂的正则表达式分割文本，保留有意义的词语
    2. 过滤短词和将单词转换为小写
    3. 处理特殊符号和数字
    """
    # 替换特殊符号为空格
    text = re.sub(r'[^\w\s]', ' ', bigString)
    # 分割文本为单词列表
    listOfTokens = re.split(r'\s+', text)
    # 过滤短词，转换为小写
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet|set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print(f"the word:{word} is not in my Vocabulary!")
    return returnVec

def bagOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)                           
    numWords = len(trainMatrix[0])                           
    pAbusive = sum(trainCategory)/float(numTrainDocs)        
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)    
    p0Denom = 2.0; p1Denom = 2.0                
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:                            
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:                                                
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)                            
    p0Vect = np.log(p0Num/p0Denom)        
    return p0Vect,p1Vect,pAbusive                           

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify*p1Vec)+np.log(pClass1)
    p0 = sum(vec2Classify*p0Vec)+np.log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0
    

def spamTest():
    docList=[]
    classList=[]
    fullText=[]
    for i in range(1, 26):  # 遍历25个txt文件
        try:
            # 修正文件路径 - 使用正确的路径 'Bayes/spam' 而不是 'email/spam'
            with open(f'spam/{i}.txt', 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                wordList = textParse(content)  # 读取每个垃圾邮件，并字符串转换成字符串列表
                docList.append(wordList)
                classList.append(1)  # 标记垃圾邮件，1表示垃圾文件
                print(f"已处理垃圾邮件 {i}.txt，提取了 {len(wordList)} 个词")
                # 打印前10个解析后的词语，用于检查解析效果
                if i == 1:
                    print(f"第一个垃圾邮件的前10个词: {wordList[:10]}")
        except Exception as e:
            print(f"处理垃圾邮件 {i}.txt 时出错: {e}")

        try:
            # 同样修正非垃圾邮件的路径
            with open(f'ham/{i}.txt', 'r', encoding='latin-1', errors='ignore') as f:
                content = f.read()
                wordList = textParse(content)  # 读取每个非垃圾邮件，并字符串转换成字符串列表
                docList.append(wordList)
                classList.append(0)  # 标记非垃圾邮件，0表示非垃圾文件
                print(f"已处理非垃圾邮件 {i}.txt，提取了 {len(wordList)} 个词")
        except Exception as e:
            print(f"处理非垃圾邮件 {i}.txt 时出错: {e}")
    
    vocabList = createVocabList(docList)  # 创建词汇表，不重复
    trainingSet=list(range(50))
    testSet=[]
    for i in range(10):
        randIndex=int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]
    trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam=trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount=0
    for docIndex in testSet:
        wordVector=setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1
    print(f'错误率: {float(errorCount)/len(testSet)}')
    return float(errorCount)/len(testSet)

if __name__ == '__main__':
    spamTest()

