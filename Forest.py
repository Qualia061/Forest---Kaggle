# -*- coding: utf-8 -*-
"""
Created on Fri May 18 16:45:05 2018

@author: hayas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#loading the data
train = pd.read_csv('E:/Python Github/Forest---Kaggle/train.csv')
test = pd.read_csv('E:/Python Github/Forest---Kaggle/test.csv')

#Observing the data
train.info()
train.describe()
test.info()
test.describe()

sns.heatmap(train.drop("Id", axis=1).corr(), linecolor='white', annot=True,vmin=0, vmax=1)

train_noareasoil=train.iloc[:,1:11]
train_noareasoil=pd.concat([train_noareasoil,train['Cover_Type']],axis=1)
for col in train_noareasoil:
    plt.figure()
    print(sns.distplot(train_noareasoil[col]))
   
corrDf = train_noareasoil.corr() 
corrDf['Cover_Type'].sort_values(ascending =False)

#Visualization
train_noId=train.iloc[:,1:]
cols = train_noId.columns
size = len(cols)-1
x = cols[size]
y = cols[0:size]
for i in range(0,size):
    sns.violinplot(data=train_noId,x=x,y=y[i])  
    plt.show()




#Split the training set
source_X=train.iloc[:,1:55]
source_y=train.loc[:,'Cover_Type']

from sklearn.cross_validation import train_test_split
train_X, test_X, train_y, test_y = train_test_split(source_X,source_y,train_size=.8)

#Tuning the parameters
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV

gbr = GradientBoostingClassifier(learning_rate=0.1)
gbr.fit( train_X , train_y )
gbr.score(test_X , test_y )
print(gbr.score(test_X , test_y ))

params={'n_estimators':[x for x in range(40,60,10)]}
gbr_best = GradientBoostingClassifier(n_estimators=60,learning_rate=0.2,min_samples_split = 90,min_samples_leaf = 30,max_depth = 8,max_features = 14,subsample = 0.85)
grid = GridSearchCV(gbr_best, params, cv=5, scoring="r2")
grid.fit( source_X , source_y )
grid.grid_scores_
grid.best_estimator_

gbr_best = GradientBoostingClassifier(n_estimators=500,learning_rate=0.2,min_samples_split = 90,min_samples_leaf = 30,max_depth = 8,max_features = 14,subsample = 0.85)
gbr_best.fit( train_X , train_y )
gbr_best.score(test_X , test_y )
print(gbr_best.score(test_X , test_y))

#Alternative model
from xgboost.sklearn import XGBClassifier
from sklearn.grid_search import GridSearchCV

param_test = {
 'reg_alpha':[1e-5, 5e-5,1e-4,5e-4, 1e-3]
}
xgb_best = XGBClassifier(
 learning_rate =0.1,
 n_estimators=150,
 max_depth=10,
 min_child_weight=1,
 gamma=0.1,
 subsample=0.9,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27,
 reg_alpha=5e-5
 )

grid = GridSearchCV(estimator = xgb_best, param_grid = param_test, cv=5)
grid.fit( source_X , source_y )
grid.grid_scores_
grid.best_estimator_

xgb_best=grid.best_estimator_
xgb_best.fit( train_X , train_y )
xgb_best.score(test_X , test_y )
print(xgb_best.score(test_X , test_y))

#Another model
from sklearn.ensemble import RandomForestClassifier

params={'min_samples_leaf':[x for x in range(1,20,2)]}
rf_best = RandomForestClassifier(n_estimators=130,max_features=18,n_jobs=-1,min_samples_leaf=1)
grid = GridSearchCV(rf_best, params, cv=5)
grid.fit( source_X , source_y )
grid.grid_scores_
grid.best_estimator_


rf_best2=grid.best_estimator_
rf_best2.fit( train_X , train_y )
rf_best2.score(test_X , test_y )
print(rf_best2.score(test_X , test_y ))




#Make prediction and save the csv file

model=XGBClassifier(
 learning_rate =0.1,
 n_estimators=150,
 max_depth=10,
 min_child_weight=1,
 gamma=0.1,
 subsample=0.9,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 reg_alpha=5e-5
 )


model.fit( source_X,source_y)
pred_X=test.iloc[:,1:]
pred_y = model.predict(pred_X)

submission = pd.DataFrame({"Id": test["Id"], "Cover_Type": pred_y})
submission=submission[["Id","Cover_Type"]]
submission.shape
print(submission.head())
submission.to_csv("E:/Python Github/Forest---Kaggle/submission.csv", index=False)


















