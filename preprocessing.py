#!/usr/bin/python
# -*- coding: UTF-8 -*-

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
import os
from imblearn.combine import SMOTEENN
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing

#获取训练集、测试集
def get_data():
	ad_feature = pd.read_csv('./data/adFeature.csv')
	if os.path.exists('./data/userFeature.csv'):
		user_feature = pd.read_csv('./data/userFeature.csv')
	else:
		userFeature_data = []
		with open('./data/userFeature.data', 'r') as f:
			for i, line in enumerate(f):
				line = line.strip().split('|')
				userFeature_dict = {}
				for each in line:
					each_list = each.split(' ')
					userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
				userFeature_data.append(userFeature_dict)
				if i % 100000 == 0:
					print(i)
			print("开始写入dataframe...")
			user_feature = pd.DataFrame(userFeature_data)
			print("开始写入文件：...")
			user_feature.to_csv('./data/userFeature.csv', index=False)
			#userFeature.csv格式，LBS地理位置
			#LBS,age,appIdAction,appIdInstall,carrier,consumptionAbility,ct,education,gender,house,interest1,interest2,interest3,interest4,interest5,kw1,kw2,kw3,marriageStatus,os,topic1,topic2,topic3,uid
			#950,4,,,1,2,3 1,7,2,,93 70 77 86 109 47 75 69 45 8 29 49 83 6 46 36 11 44 30 118 76 48 28 106 59 67 41 114 111 71 9,46 19 13 29,,,52 100 72 131 116 11 71 12 8 113 28 73 6 132 99 76 46 62 121 59 129 21 93,664359 276966 734911 103617 562294,11395 79112 115065 77033 36176,,11,2,9826 105 8525 5488 7281,9708 5553 6745 7477 7150,,26325489

	train = pd.read_csv('./data/train.csv')#返回一个dataFrame
	predict = pd.read_csv('./data/test1.csv')
	train.loc[train['label'] == -1, 'label'] = 0 #"label"列中=-1的换成0，分类变成0，1两类（之前是-1，1）
	predict['label'] = -1#predict 的label都是-1
	data = pd.concat([train, predict])#竖着连接
	data = pd.merge(data, ad_feature, on='aid', how='left')#how:"left"只保留左表的所有数据，即对齐时要保证左表数据全在，右表数据用来补充
	data = pd.merge(data, user_feature, on='uid', how='left')
	data = data.fillna('-1')

	return data#data的格式是：aid,uid,label,ad_feature,user_feature

#X是一个二维数组，每一行是一个实例的特征
def remove_low_variance_feature(X, varThreshold):
	selector = VarianceThreshold(varThreshold)#不加参数时默认是0.0
	selector.fit(X)
	retained_index = selector.get_support(True)#the array of int which is the index of retained feature in X
	print("after remove_low_variance_feature, the retained features index is")
	print retained_index

#data is dataFrame, just one_hot_feature and label
#feature和label之间的关系不好散点图表示（二分类），应该画出每个类里的特征的分布（bins）
def plot_show(data):
	labels = data['label']
	col_list = data.columns.values.tolist()
	for i in range(9, len(col_list)):#0 is label
		for j in range(i+1, len(col_list)):
			plt.plot(np.array(data[col_list[i]]), np.array(data[col_list[j]]), '.')
			plt.title(col_list[i] + ' VS ' + col_list[j])
			plt.show()
			#plt.savefig('Figure_' + str(i) + '_' + str(j) + '.png')

#特征直方图展示		
def plot_show2(data):
	data0 = data[data.label == 0]
	data1 = data[data.label == 1]
	for col in data.columns.values.tolist():
		if col == 'label':
			continue
		plt.subplot(121)
		plt.hist(np.array(data0[col], dtype=np.int64))
		plt.title(col + ' label 0')
		plt.subplot(122)
		plt.hist(np.array(data1[col], dtype=np.int64))
		plt.title(col + ' label 1')
		plt.show()
		##plt.savefig('Figure_label_' + col + '.png')

#不同类别样本数目计算
def cal(data):
	data0 = data[data.label == 0]
	data1 = data[data.label == 1]
	print("label 0 %d" % data0.iloc[:,0].size)
	print("label 1 %d" % data1.iloc[:,0].size)
	
#卡方检验
#data 只有one-hot feature and label
def kafang(data):
	data0 = data[data.label == 0]
	data1 = data[data.label == 1]
	label0 = np.array(data0['label'])
	label1 = np.array(data1['label'])
	X0 = data0.drop('label', axis = 1)
	X1 = data1.drop('label', axis = 1)
	age0 = data0['age']
	age1 = data1['age']
	label = data['label']
	X = data.drop('label', axis = 1)
	selectModel = SelectKBest(chi2, k='all')
	selectModel.fit(np.array(X), np.array(label))
	print("chi2 scores")
	print(selectModel.scores_)
	print("chi2 p_values")
	print(selectModel.pvalues_)
	
	
	
#过采样	
def over_sampling(data):
	data = data.drop('aid', axis = 1)
	data = data.drop('uid', axis = 1)
	y = data['label']
	X = data.drop('label', axis = 1)
	sme = SMOTEENN()
	X_res, y_res = sme.fit_sample(X, y)
	data_res = pd.concat([X_res, y_res], axis = 1)
	data_res.to_csv('./data/train_all_after_overSamlping.csv', index = False)
	
if __name__ == '__main__':
	data = get_data()
	#testData
	#data = pd.read_csv('./data/testData.csv')
	print("before operation  ")
	 # 训练集
	train = data[data.label != -1]
	test = data[data.label == -1]
	#over_sampling(train)
	
	#print data
	one_hot_features = train #one_hot_feature dataFrame
	#one_hot 只有一个值
	one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
					   'marriageStatus', 'advertiserId', 'campaignId', 'creativeId',
					   'adCategoryId', 'productId', 'productType', 'creativeSize']#17 个 feature
	vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5',
					  'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
	print("one_hot_feature LabelEncoder >>>>>>>>>>>")
	for feature in vector_feature:
		print("feature %s" %feature)#对于每个feature,比如LBS,所有取值是[100,200,123],encoder之后变成编号0,1,2(顺序不一定)
		one_hot_features = one_hot_features.drop(feature, axis=1)
	#for marriageStatus and ct which are one_hot_features, but they have more than one feature,so drop it
	one_hot_features = one_hot_features.drop('marriageStatus', axis=1)
	one_hot_features = one_hot_features.drop('ct', axis=1)
	one_hot_features = one_hot_features.drop('os', axis=1)
	one_hot_features = one_hot_features.drop('aid', axis=1)
	one_hot_features = one_hot_features.drop('uid', axis=1)
	#kafang(one_hot_features)
	#plot_show(one_hot_features)
	#plot_show2(one_hot_features)
	#cal(one_hot_features)
	age = np.array(one_hot_features['age'], dtype=np.int).reshape(-1,1)
	hou = np.array(one_hot_features['house'], dtype=np.int)
	chiModel = chi2(age, hou)
	print("chi2 result between age and house")
	print(chiModel)
	#plt.plot(age, hou, '.r', alpha=0.1, ms=3)
	#plt.title('age house')
	#plt.show()
	#plt.plot(np.array(one_hot_features['age']), np.array(one_hot_features['consumptionAbility']), '.r', alpha=0.1, ms=10)
	#plt.title('age consumptionAbility')
	#plt.show()
	#one_hot_features = one_hot_features.drop('label', axis=1)
	
	
	#print("after drop vector feature ")
	#print one_hot_features

	#remove_low_variance_feature(np.array(one_hot_features), 0.01)#no remove

