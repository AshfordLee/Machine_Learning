from sklearn.linear_model import LogisticRegression
import os

def colicSklearn_liblinear():
    current_file=os.path.dirname(__file__)
    train_file_path=os.path.join(current_file,'horseColicTraining.txt')
    test_file_path=os.path.join(current_file,'horseColicTest.txt')
    train_file=open(train_file_path,'r')
    test_file=open(test_file_path,'r')
    training_set=[]
    training_labels=[]
    test_set=[]
    test_labels=[]
    
    for line in train_file.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        training_set.append(lineArr)
        training_labels.append(float(currLine[-1]))

    for line in test_file.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        test_set.append(lineArr)
        test_labels.append(float(currLine[-1]))

    classifier=LogisticRegression(solver='liblinear',
                                  max_iter=100).fit(training_set,training_labels)
    test_accurcy=classifier.score(test_set,test_labels)
    print("the test accuracy is:%f"%test_accurcy)


def colicSklearn_sag():
    current_file=os.path.dirname(__file__)
    train_file_path=os.path.join(current_file,'horseColicTraining.txt')
    test_file_path=os.path.join(current_file,'horseColicTest.txt')
    train_file=open(train_file_path,'r')
    test_file=open(test_file_path,'r')
    training_set=[]
    training_labels=[]
    test_set=[]
    test_labels=[]
    
    for line in train_file.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        training_set.append(lineArr)
        training_labels.append(float(currLine[-1]))

    for line in test_file.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        test_set.append(lineArr)
        test_labels.append(float(currLine[-1]))

    classifier=LogisticRegression(solver='sag',
                                  max_iter=100).fit(training_set,training_labels)
    test_accurcy=classifier.score(test_set,test_labels)
    print("the test accuracy is:%f"%test_accurcy)

if __name__=='__main__':
    colicSklearn_liblinear()
    colicSklearn_sag()