import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import catboost as cb
from xgboost import XGBRegressor
from mlxtend.regressor import StackingCVRegressor #pip installed!
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.svm import SVR

#this is my feat engineering with Lim's ensemble modelling methods
#Its error equals 0.12931 cf. 0.12286 when done with Lim's engineering
#see: https://www.kaggle.com/code/limyenwee/stacked-ensemble-models-top-3-on-leaderboard/notebook

#reacquaint with data
train_set = pd.read_csv('train.csv', header=0)
test_set = pd.read_csv('test.csv', header=0)
housing = pd.concat([train_set, test_set])

#calc & plot prop diff btwn train and test means
diffs = train_set.describe().iloc[1, :] / test_set.describe().iloc[1, :]
plt.scatter(range(diffs.size), diffs)

#calc ci & id outlier means btwn train & test
from scipy.stats import norm
ci = norm(*norm.fit(diffs.dropna())).interval(0.95)  # fit a normal distribution and get 95% c.i.
print(diffs[(diffs > ci[1]) | (diffs < ci[0])]) # = '3SsnPorch', 'Id', 'LowQualFinSF', 'PoolArea'

#rm weird feats
tbr = diffs[(diffs > ci[1]) | (diffs < ci[0])].index
for rm in tbr: del housing[rm]

#find object/categorical features
objs = np.array([housing.dtypes, housing.columns]) #3 dtypes
obj_feats = objs[1][objs[0] == 'O']
norm_feats = objs[1][objs[0] == 'float64']
int_feats = objs[1][objs[0] == 'int64'] #some are areas

to_encode = ['MSSubClass'] + obj_feats.tolist()
housing = housing.reset_index(drop=True)

for enc in to_encode:
    cat_encoder = OneHotEncoder()
    housing_cat_1hot = cat_encoder.fit_transform(housing[[enc]])

    housing = housing.drop(enc, axis=1)

    housing = pd.concat([housing, pd.DataFrame(housing_cat_1hot.toarray())], axis=1)
    housing.columns = list(housing.columns[:(housing_cat_1hot.shape[1] * -1)]) + [f"{enc}_{qw}" for qw in range(housing_cat_1hot.shape[1])]


#impute missing vals for numerical
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
cols = housing.columns.tolist()
cols.remove('SalePrice')

imputer.fit(housing[cols])
X = imputer.transform(housing[cols])
housing_tr = pd.DataFrame(X, columns=cols, index=housing.index)

housing = pd.concat([housing_tr, housing.SalePrice], axis=1)

#resample training & test data
train_x = housing.iloc[:train_set.shape[0], 1:-1]
train_y = housing.SalePrice[:train_set.shape[0]]
test_x = housing.iloc[train_set.shape[0]:, 1:-1]

#scale training & test data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_x)
X = scaler.transform(train_x)
train_x = pd.DataFrame(X, columns=train_x.columns, index=train_x.index)

scaler.fit(test_x)
X = scaler.transform(test_x)
test = pd.DataFrame(X, columns=test_x.columns, index=test_x.index)


#NB I use logs here - must relate to  np.exp() use on Lim's imputation!!!!!!!!!!!
train_y = np.log(train_y)
full = pd.concat([train_x, train_y], axis=1)
dev_train, dev_test = train_test_split(full, test_size=0.2, shuffle=True)
dev_train_y = dev_train['SalePrice']
dev_train_X = dev_train.drop(['SalePrice'], axis=1)
dev_test_y = dev_test['SalePrice']
dev_test_X = dev_test.drop(['SalePrice'], axis=1)

ridgemodel = Ridge(alpha=26)
xgbmodel = XGBRegressor(alpha=3, colsample_bytree=0.5, reg_lambda=3, learning_rate=0.01,\
           max_depth=3, n_estimators=10000, subsample=0.65)
svrmodel = SVR(C=8, epsilon=0.00005, gamma=0.0008)
hubermodel = HuberRegressor(alpha=30, epsilon=3, fit_intercept=True, max_iter=2000)
cbmodel = cb.CatBoostRegressor(loss_function='RMSE', colsample_bylevel=0.3, depth=2, \
          l2_leaf_reg=20, learning_rate=0.005, n_estimators=15000, subsample=0.3,verbose=False)
stackmodel = StackingCVRegressor(regressors=(ridgemodel, xgbmodel, svrmodel, hubermodel, cbmodel),
             meta_regressor=cbmodel, use_features_in_secondary=True)

import warnings
warnings.filterwarnings("ignore")
import time

start = time.time() #takes 9mins
print("Recording Modelling Time")
for i in [ridgemodel, hubermodel, cbmodel, svrmodel, xgbmodel, stackmodel]:
    i.fit(train_x, train_y)
    if i == stackmodel:
        i.fit(np.array(dev_train_X), np.array(dev_train_y))
end = time.time()
print("Time Elapsed: " + str(round((end - start)/60,0)) +"minutes.")



print("Overview of model performance")
for i in [ridgemodel, hubermodel, cbmodel, svrmodel, xgbmodel, stackmodel]:
    print("\n")
    #print(i)
    print("RMSLE of Development train set: ")
    print(mean_squared_error(dev_train_y,i.predict(dev_train_X), squared=False))
    print("RMSLE of Development test set: ")
    print(mean_squared_error(dev_test_y,i.predict(dev_test_X), squared=False))
    print("\n")
print("-----------------------------")
print("RMSLE of Development train set using ensemble model: ")
fit = (svrmodel.predict(train_x) + xgbmodel.predict(train_x) +   stackmodel.predict(train_x) + ridgemodel.predict(train_x) + hubermodel.predict(train_x) + cbmodel.predict(train_x)) / 6
print(mean_squared_error(train_y, fit, squared=False))



#This time we fit the models with all the data.
start = time.time()
print("Recording Modelling Time")
for i in [ridgemodel, hubermodel, cbmodel, svrmodel, xgbmodel, stackmodel]:
    i.fit(train_x, train_y)
    if i == stackmodel:
        i.fit(np.array(train_x), np.array(train_y))
end = time.time()
print("Time Elapsed: " + str(round((end - start)/60,0)) +"minutes.")


final_prediction = (np.exp(ridgemodel.predict(test)) + 3 * np.exp(xgbmodel.predict(test)) \
+  5 * np.exp(stackmodel.predict(test)) + 4 * np.exp(svrmodel.predict(test)) \
+  np.exp(hubermodel.predict(test)) +  np.exp(cbmodel.predict(test))) / 15

submission = pd.DataFrame(final_prediction, index = test.index)
submission.reset_index(drop=False, inplace = True)
submission = submission.rename(columns={0 : 'SalePrice', 'index' : 'Id'})
submission.Id = submission.Id + 1 #Id's got scrambled!!
submission.to_csv('MyFeatEng_LimModelEnsemble.csv', index=False) # Score 0.12931 on 13/04/2022

