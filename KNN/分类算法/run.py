#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd


# In[26]:


# 读取鸢尾花数据集 header 参数指定标题行 默认为0 ， 若无标题则 header = None
data = pd.read_csv(r"iris.csv",header = 0)
# 获取数据的简单说明
# =========================
# 显示前 n 条记录 默认 n = 5
# data.head(10)
# 显示末尾 n 条记录  默认 n = 5
# data.tail()
# 随机抽取 n 条记录 默认 n = 1 
# data.sample(10)
# 查看数据集记录数
# len(data)
# ===========================

# 对文本进行格式处理
# ============================================
# 将类别文本映射成数值类型 
data["species"] = data["species"].map({"setosa":0,"virginica":1,"versicolor":2 })
# 删除不需要的列, axix 0 表示行 1 表示列 inplace 表示在原有的数据集中更新
# 等同于 data = data.drop("sepal_length",axis = 1)
#data.drop("sepal_length",axis = 1,inplace = True)
# .any（） 表示，若有重复的记录则返回 true
#data.duplicated().any()
# 删除重复的记录 
data.drop_duplicates(inplace = True)
# 长各个类别的鸢尾花具有多少条记录
# data["species"].value_counts()
# data.sample(10)


# In[63]:
class KNN:
	def __init__(self,k):
		self.k = k 
					        
	def fit(self, X, y):
		self.X = np.asarray(X)
		self.y = np.asarray(y)
										    
	def predict(self,X):
		X = np.asarray(X)
		result = []
		for x in X :
			dis = np.sqrt(np.sum((x - self.X) ** 2 , axis = 1))
			index = dis.argsort()
			index = index[:self.k]
			count = np.bincount(self.y[index])
			result.append(count.argmax())
		return np.asarray(result)

	        


# In[70]:


# 提取每个类别的鸢尾花数据
t0 = data[data["species"] == 0]
t1 = data[data["species"] == 1]
t2 = data[data["species"] == 2]
# 打乱训练集的记录条目,对每个类别的数据进行洗牌
t0 = t0.sample(len(t0),random_state=0)
t1 = t1.sample(len(t1),random_state=0)
t2 = t2.sample(len(t2),random_state=0)
# 将记录区分为训练集与数据集,按照纵向的方式，将t0-t2中前40项组合拼接
train_X = pd.concat([t0.iloc[:40,:-1],t1.iloc[:40,:-1],t2.iloc[:40,:-1]],axis = 0)
train_y = pd.concat([t0.iloc[:40,-1],t1.iloc[:40,-1],t2.iloc[:40,-1]],axis = 0)
# 测试集 同训练集
test_X = pd.concat([t0.iloc[40:,:-1],t1.iloc[40:,:-1],t2.iloc[40:,:-1]],axis = 0)
test_y = pd.concat([t0.iloc[40:,-1],t1.iloc[40:,-1],t2.iloc[40:,-1]],axis = 0)
# 创建 KNN 对象 ，进行训练与测试
knn = KNN(k=3)
# 进行训练
knn.fit(train_X,train_y)
# 进行测试，获得测试结果
result = knn.predict(test_X)
# display(result)
# display(test_y)

# display(np.sum(result == test_y))
# display(np.sum(result == test_y) / len(result))


# In[72]:


import matplotlib as mpl
import matplotlib.pyplot as plt


# In[ ]:


# 默认情况下 matplotlib 不支持中文显示，设置任意支持中文的字体即可
mpl.rcParams["font.family"] = "WenQuanYi Micro Hei"
# 设置中文字体中正常显示负号（-）。即不适用 unicode 中的 “-” 展示
# mpl.rcParams["axes.unicode_minux"] = False


# In[89]:


# "setosa":0,"virginica":1,"versicolor":2 
# 设置画布大小
plt.figure(figsize=(10,10))
# 绘制训练集中的数据
plt.scatter(x = t0["sepal_length"][:40],y = t0["petal_length"][:40],color = "r" , label = "setosa")
plt.scatter(x = t1["sepal_length"][:40],y = t1["petal_length"][:40],color = "g" , label = "virginica")
plt.scatter(x = t2["sepal_length"][:40],y = t2["petal_length"][:40],color = "b" , label = "versicolor")

# 绘制测试集中的数据
# 获取所有预测正确的值
right = test_X[result == test_y]
# 获取所有预测错误的值
wrong = test_X[result != test_y]
plt.scatter(x = right["sepal_length"],y = right["petal_length"],color = "c" ,marker = "x",label = "right")
plt.scatter(x = wrong["sepal_length"],y = wrong["petal_length"],color = "m" ,marker = ">",label = "wrong")
# 设置坐标轴
plt.xlabel("花萼长度")
plt.ylabel("花瓣长度")
plt.title("KNN 分类结果显示")
# 设置图例
plt.legend(loc="best")
#plt.show()


