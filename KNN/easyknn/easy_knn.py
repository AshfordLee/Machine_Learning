import numpy as np
import operator
import collections

def createDataSet():
    group=np.array([[1,101],[5,89],[108,5],[115,8]])
    labels=['爱情片','爱情片','动作片','动作片']

    return group, labels

def classify0(inx,dataset,labels,k):
    dist=np.sum((inx-dataset)**2,axis=1)**0.5
    k_labels=[labels[index] for index in dist.argsort()[0:k]]
    label=collections.Counter(k_labels).most_common(1)[0][0]

    return label



if __name__=='__main__':
    group,labels=createDataSet()
    test=[101,20]
    test_class=classify0(test,group,labels,k=3)
    print(f"预测类别为：{test_class}")