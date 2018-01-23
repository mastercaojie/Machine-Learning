# -*- coding: utf-8 -*-
import numpy as np  # numpy库
from sklearn import cross_validation
from sklearn.model_selection import cross_val_score  # 交叉检验
import pandas as pd  # 导入pandas
from sklearn.ensemble import RandomForestClassifier


#获取数据
data = pd.read_csv('corrected_new.csv',header=None,delimiter=",")
dataset = np.array(data)
print("数据集shape: ",dataset.shape)
print (70 * '-')  # 打印分隔线
X = dataset[:,0:35]
Y = dataset[:,35]
Label = np.array(['0','1','2','3','4'])
print("输出类别:",Label)
print (70 * '-')  # 打印分隔线
# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.1)
print("训练集个数：",X_train.shape[0])
print("测试集个数：",X_test.shape[0])
print (70 * '-')  # 打印分隔线
#用随机森林
rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
rf.fit(X_train,Y_train)
probability = pd.DataFrame(np.array(rf.predict_proba(X_test)))
result = pd.DataFrame(np.array((rf.predict(X_test))))
score = cross_val_score(rf, X_train,Y_train)
print("训练集精度得分:",score.mean())
score = cross_val_score(rf, X_test,Y_test)
print("测试集精度得分:",score.mean())
showlist = pd.concat([probability,result],axis=1)
print (70 * '-')  # 打印分隔线
print("概率+类别：")
showlist.columns = ['类别0','类别1','类别2','类别3','类别4',"结果"]
prob0 = showlist[showlist['结果'].isin([0])].shape[0]
prob1 = showlist[showlist['结果'].isin([1])].shape[0]
prob2 = showlist[showlist['结果'].isin([2])].shape[0]
prob3 = showlist[showlist['结果'].isin([3])].shape[0]
prob4 = showlist[showlist['结果'].isin([4])].shape[0]
resultlist =[prob0,prob1,prob2,prob3,prob4]
resultlist = pd.DataFrame(resultlist)
print(showlist)
print (70 * '-')  # 打印分隔线
print("统计每个类别的数量：")
resultlist.columns = ['统计']
print(resultlist)

