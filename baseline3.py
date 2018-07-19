# coding=utf-8
# @author:wuyy
# blog: https://blog.csdn.net/bryan__
#
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
from sklearn.cross_validation import cross_val_score
import os
from sklearn.metrics import accuracy_score

ad_feature=pd.read_csv('../data/adFeature.csv')
if os.path.exists('../data/userFeature.csv'):
    user_feature=pd.read_csv('../data/userFeature.csv')
else:
    userFeature_data = []
    with open('../data/userFeature.data', 'r') as f:
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
        user_feature.to_csv('../data/userFeature.csv', index=False)


train=pd.read_csv('../data/train.csv')
predict=pd.read_csv('../data/test1.csv')
train.loc[train['label']==-1,'label']=0
predict['label']=-1
data=pd.concat([train,predict])
data=pd.merge(data,ad_feature,on='aid',how='left')
data=pd.merge(data,user_feature,on='uid',how='left')
data=data.fillna('-1')
one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus','advertiserId','campaignId', 'creativeId',
       'adCategoryId', 'productId', 'productType','creativeSize']
vector_feature=['appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']
for feature in one_hot_feature:
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])

train=data[data.label!=-1]
train_y=train.pop('label')
# train, test, train_y, test_y = train_test_split(train,train_y,test_size=0.2, random_state=2018)
test=data[data.label==-1]
res=test[['aid','uid']]
test=test.drop('label',axis=1)
enc = OneHotEncoder()
#train_x=train[['creativeSize']]
#test_x=test[['creativeSize']]
train_x=None
test_x=None

for i,feature in enumerate(one_hot_feature):
    enc.fit(data[feature].values.reshape(-1, 1))
    train_a=enc.transform(train[feature].values.reshape(-1, 1))
    test_a = enc.transform(test[feature].values.reshape(-1, 1))
    if i==0:
        train_x=train_a
        test_x=test_a
    else:
        train_x= sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))
print('one-hot prepared !')

cv=CountVectorizer()
for feature in vector_feature:
    cv.fit(data[feature])
    train_a = cv.transform(train[feature])
    test_a = cv.transform(test[feature])
    train_x = sparse.hstack((train_x, train_a))
    test_x = sparse.hstack((test_x, test_a))
print('cv prepared !')

def LGB_test(train_x,train_y,test_x,test_y):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=1500, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50,random_state=2018,n_jobs=-1,class_weight={'class_label':'scale_pos_weight'},
    )
    clf.fit(train_x, train_y,eval_set=[(train_x, train_y),(test_x,test_y)],eval_metric='auc',early_stopping_rounds=100)
    # print(clf.feature_importances_)
    return clf,clf.best_score_[ 'valid_1']['auc']

def LGB_predict(train_x,train_y,test_x,res):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=40, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=2200, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50, random_state=2020
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc',early_stopping_rounds=100)
    res['score'] = clf.predict_proba(test_x)[:,1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv('../data/submission.csv', index=False)
    os.system('zip ../data/baseline.zip ../data/submission.csv')
    return clf


#选出最优参数
train_xx, test_xx, train_yy, test_yy = train_test_split(train_x, train_y, test_size=0.2, random_state=0)

def LGB_args(args):
    max_depth = args["max_depth"] + 5
    num_leaves=args["num_leaves"]* 5 +30
    learning_rate=args["learning_rate"]*0.01+0.04
    min_child_weight = args["min_child_weight"]*10 + 50
    subsample = args["subsample"] * 0.1 + 0.7
    reg_alpha=args['reg_alpha']
    reg_lambda = args['reg_lambda']

    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=num_leaves, reg_alpha=reg_alpha, reg_lambda=reg_lambda,
        max_depth=max_depth, n_estimators=200, objective='binary',
        subsample=subsample, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=learning_rate, min_child_weight=min_child_weight#, random_state=2020
    )

    metric = cross_val_score(clf, train_x,  train_y, cv=5, scoring="roc_auc").mean()
    print("metric:>>>",metric)
    return -metric
    '''
    clf.fit(train_xx, train_yy, eval_set=[(train_xx, train_yy)], eval_metric='auc', early_stopping_rounds=100)
    res=clf.predict_proba(test_xx)[:,1]
    return -accuracy_score(res, test_yy)
    '''

from hyperopt import fmin,tpe,hp,partial
space = {"num_leaves":hp.randint("num_leaves",10),
         "max_depth":hp.randint("max_depth",15),
          "learning_rate":hp.randint("learning_rate",6),
         "reg_alpha":hp.uniform("reg_alpha",0,1),
         "reg_lambda":hp.uniform("reg_lambda",0,1),
         "subsample":hp.randint("subsample",4),
          "min_child_weight":hp.randint("min_child_weight",5),
         }
algo = partial(tpe.suggest,n_startup_jobs=10)
best = fmin(LGB_args,space,algo = algo,max_evals=4)
print (best)

