# -*- coding = utf-8 -*-
# @time:2020/9/23 8:20
# Author:TC
# @File:Decision Tree.py
# @Software:PyCharm


from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pydotplus
from six import StringIO


wine=load_wine()
# print(wine.data)
print(wine.data.shape)
# print(wine.target)
print(wine.target.shape)

# import pandas as pd
# df=pd.concat([pd.DataFrame(wine.data),pd.DataFrame(wine.target)])
# print(df.head())

print(wine.feature_names)
print(wine.target_names)

Xtrain,Xtest,Ytrain,Ytest=train_test_split(wine.data,wine.target,test_size=0.3)
print(Xtrain.shape)
print(Xtest.shape)

clf=tree.DecisionTreeClassifier()
clf=clf.fit(Xtrain,Ytrain)
score=clf.score(Xtest,Ytest)
print(score)

import graphviz

# dot_data = StringIO()
# tree.export_graphviz(clf, out_file=dot_data,
#                          feature_names=wine.feature_names,
#                          class_names=wine.target_names,
#                          filled=True, rounded=True,
#                          special_characters=True)
#
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_pdf("iris1.pdf")  # 写入pdf

dot_data = tree.export_graphviz(clf, out_file=None,
    feature_names=wine.feature_names,
    class_names=wine.target_names,
    filled=True, rounded=True,
    special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris.pdf")