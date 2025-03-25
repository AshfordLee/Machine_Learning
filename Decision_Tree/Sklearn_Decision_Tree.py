from sklearn import tree
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from six import StringIO
import pydotplus

if __name__ == "__main__":
    # 使用绝对路径打开文件
    # 获取当前脚本的绝对路径，然后计算lenses.txt的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'lenses.txt')
    fr = open(file_path)
    lenses_target=[]

    lensesLabels=['age','prescript','astigmatic','tearRate']
    lenses=[inst.strip().split('\t') for inst in fr.readlines()]
    for each in lenses:
        lenses_target.append(each[-1])

    lenses_list=[]   
    lenses_dict={}
    for each_label in lensesLabels:
        for each in lenses:
            lenses_list.append(each[lensesLabels.index(each_label)])

        lenses_dict[each_label]=lenses_list
        lenses_list=[]
    

    lenses_pd=pd.DataFrame(lenses_dict)
    le=LabelEncoder()
    for col in lenses_pd.columns:
        lenses_pd[col]=le.fit_transform(lenses_pd[col])

    clf=tree.DecisionTreeClassifier()
    lenses=clf.fit(lenses_pd.values.tolist(),lenses_target)
    dot_data=StringIO()
    tree.export_graphviz(
        clf,
        out_file=dot_data,
        feature_names=lenses_pd.keys(),
        class_names=clf.classes_,
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("tree.pdf")

    print(clf.predict([[1,1,1,0]]))
    # print(lenses_pd)