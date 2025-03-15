import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib
import operator
import os
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
    
    
if __name__=="__main__":
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建数据文件的完整路径
    data_file = os.path.join(script_dir, 'datingTestSet.txt')
    # 使用完整路径
    datingDataMat,datingLabels=file_to_matrix(data_file)
    showdatas(datingDataMat,datingLabels)