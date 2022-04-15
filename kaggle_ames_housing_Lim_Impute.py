import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
sns.set_style('darkgrid')
from scipy.stats import skew, norm, probplot

import time
from sklearn.preprocessing import OneHotEncoder, RobustScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge, HuberRegressor, LinearRegression
from sklearn.svm import SVR
from sklearn.cluster import KMeans
import catboost as cb
from xgboost import XGBRegressor
from mlxtend.regressor import StackingCVRegressor #pip installed!
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_score

#from Yen Wee Lim: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/discussion/313689
#maybe see also: https://www.analyticsvidhya.com/blog/2020/07/types-of-feature-transformation-and-scaling/

df = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

y = df['SalePrice']
df = df.drop(['SalePrice'],axis=1)
df = df.set_index('Id')
test = test.set_index('Id')

full = pd.concat([df, test],axis=0).reset_index(drop=True)

#MSZoning gets modal vals
test.loc[(test['Neighborhood'] == 'IDOTRR') & (test['MSZoning'].isnull()), 'MSZoning'] = 'RM'
test.loc[(test['Neighborhood'] == 'Mitchel') & (test['MSZoning'].isnull()), 'MSZoning'] = 'RL'


data = full[(~full['LotFrontage'].isnull()) & (full['LotFrontage'] <= 150) & (full['LotArea'] <= 20000)]

#expect LotFrontage to be somewhat correlated with LotArea. Hence we will use LinearRegression
area_vs_frontage = LinearRegression()
area_vs_frontage_X = data['LotArea'].values.reshape(-1, 1)
area_vs_frontage_y = data['LotFrontage'].values
area_vs_frontage.fit(area_vs_frontage_X,area_vs_frontage_y)
for table in [df, test]:
    table['LotFrontage'].fillna(area_vs_frontage.intercept_ + table['LotArea'] * area_vs_frontage.coef_[0] , inplace=True)

#Alley : data description says NA means no alley access
for table in [df, test]: table['Alley'].fillna("None",inplace=True)

#Since there is only 1 data that uses NoSeWa and, we will surely fill the missing value in test set with AllPub.
#We will just drop the NoSeWa row in our training dataset since it is not found in the test set and will contribute to overfitting if left alone.
test['Utilities'].fillna("AllPub",inplace=True)

#Y NOT DROPPED HERE - BE CAREFUL!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
df.drop(df[df['Utilities'] == 'NoSeWa'].index, inplace = True)

#10 types of materials used in both the metrics. However, we can notice from the barplot that most of them are made of Vinyl. Hence, we will just fill the null values with the mode (Vinyl).
test['Exterior1st'] = test['Exterior1st'].fillna(full['Exterior1st'].mode()[0])
test['Exterior2nd'] = test['Exterior2nd'].fillna(full['Exterior2nd'].mode()[0])

#Since around 60% of our data do not have Masonry veneer. It will be used to fill the null value in row 2611 and also the other rows.
test['MasVnrType'][2611] = "BrkFace"
test['MasVnrType'] = test['MasVnrType'].fillna(full['MasVnrType'].mode()[0])
test['MasVnrArea'] = test['MasVnrArea'].fillna(0)
df['MasVnrType'] = df['MasVnrType'].fillna(full['MasVnrType'].mode()[0])
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)


#Basement Metrics : data description says BsmtFinType1 measures the Type 1 finished square feet of basement.
#However, we can see a few data in test data set having basement metrics but "0" squarefeets
for basement_metrics_cols in ['BsmtExposure','BsmtCond','BsmtQual']:
    if len(full[(full[basement_metrics_cols].notnull()) & (full['BsmtFinType1'].isnull())]) > 0 :
        print("Present with BsmtFinType1 but undetected" + basement_metrics_cols)
        display(full[(full[basement_metrics_cols].notnull()) & (full['BsmtFinType1'].isnull())])
        print("NOW")

for basement_metrics_cols in ['BsmtExposure','BsmtCond','BsmtQual']:
    if len(full[(full[basement_metrics_cols].isnull()) & (full['BsmtFinType1'].notnull())]) > 0 :
        print("\nPresent with "+ basement_metrics_cols+" but BsmtFinType1 undetected" )
        display(full[(full[basement_metrics_cols].isnull()) & (full['BsmtFinType1'].notnull())])


# We assume missing basement exposure of unfinished basement is "No".
df.loc[((df['BsmtExposure'].isnull()) & (df['BsmtFinType1'].notnull())), 'BsmtExposure'] = 'No'
test.loc[((test['BsmtExposure'].isnull()) & (test['BsmtFinType1'].notnull())), 'BsmtExposure'] = 'No'
# We impute missing basement condition with "mean" value of Typical.
test.loc[((test['BsmtCond'].isnull()) & (test['BsmtFinType1'].notnull())), 'BsmtCond'] = 'TA'
# We impute unfinished basement quality with "mean" value of Typical.
test.loc[((test['BsmtQual'].isnull()) & (test['BsmtFinType1'].notnull())), 'BsmtQual'] = 'TA'

#This test data does not have basement. Hence, those squarefeets metrics should be filled in with 0.
for square_feet_metrics in ['TotalBsmtSF','BsmtUnfSF','BsmtFinSF2','BsmtFinSF1']:
    test[square_feet_metrics][2121] = 0


#The two test data do not have basement. Hence, those bathroom amount in basement should also be filled in with 0.
for bathroom_metrics in ['BsmtFullBath','BsmtHalfBath']:
    test[bathroom_metrics][2121] = 0
    test[bathroom_metrics][2189] = 0

#The other data are assumed to not have basements hence filling in None.
for table in [df,test]:
    table[table.columns[table.columns.str.contains('Bsmt')]] = table[table.columns[table.columns.str.contains('Bsmt')]].fillna("None")

#These three metrics are safe to be filled with the mode values.

df['Electrical'].fillna('SBrkr',inplace=True)
test['Functional'].fillna('Typ',inplace=True)
test['KitchenQual'].fillna('TA',inplace=True)

#this test data do not have a garage, filling GarageArea and GarageCars with 0.

test['GarageCars'].fillna(0,inplace=True)
test['GarageArea'].fillna(0,inplace=True)

#For the SaleType column, we will impute the missing data with the mode since the mode value is kinda high too.
test['SaleType'].fillna('WD',inplace=True)


#We do not have anything extra to infer final missing columns. Hence, we will treat them as "None" which is not having those items.

df['GarageYrBlt'].fillna(0,inplace=True)
test['GarageYrBlt'].fillna(0,inplace=True)
df.fillna("None", inplace=True)
test.fillna("None", inplace=True)

df.index = df.index - 1

#see log transforms: https://www.kaggle.com/code/limyenwee?scriptVersionId=90403458&cellId=63

#Feature creation
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']

for table in [df,test]:
    table['MSSubClass'] = table['MSSubClass'].apply(str)
    table['YrSold'] = table['YrSold'].astype(str)
    table['MoSold'] = table['MoSold'].astype(str)

#some feats can be ordinally encoded
qual_dict = {'None': 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
bsmt_fin_dict = {'None': 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}

for table in [df, test]:
    table["ExterQual"] = table["ExterQual"].map(qual_dict)
    table["ExterCond"] = table["ExterCond"].map(qual_dict)
    table["BsmtQual"] = table["BsmtQual"].map(qual_dict)
    table["BsmtCond"] = table["BsmtCond"].map(qual_dict)
    table["PoolQC"] = table["PoolQC"].map(qual_dict)
    table["HeatingQC"] = table["HeatingQC"].map(qual_dict)
    table["KitchenQual"] = table["KitchenQual"].map(qual_dict)
    table["FireplaceQu"] = table["FireplaceQu"].map(qual_dict)
    table["GarageQual"] = table["GarageQual"].map(qual_dict)
    table["GarageCond"] = table["GarageCond"].map(qual_dict)
    table["BsmtExposure"] = table["BsmtExposure"].map({'None': 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}) 
    table["BsmtFinType1"] = table["BsmtFinType1"].map(bsmt_fin_dict)
    table["BsmtFinType2"] = table["BsmtFinType2"].map(bsmt_fin_dict)
    table["Functional"] = table["Functional"].map({'None': 0, "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8})
    table["GarageFinish"] = table["GarageFinish"].map({'None': 0, "Unf": 1, "RFn": 2, "Fin": 3})
    table["Fence"] = table["Fence"].map({'None': 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4})
    table["CentralAir"] = table["CentralAir"].map({'N': 0, "Y": 1})
    table["PavedDrive"] = table["PavedDrive"].map({'N': 0, "P": 1, "Y": 2})
    table["Street"] = table["Street"].map({'Grvl': 0, "Pave": 1})
    table["Alley"] = table["Alley"].map({'None': 0, "Grvl": 1, "Pave": 2})
    table["LandSlope"] = table["LandSlope"].map({'Gtl': 0, "Mod": 1, "Sev": 2})
    table["LotShape"] = table["LotShape"].map({'Reg': 0, "IR1": 1, "IR2": 2, "IR3": 3})
    
modified_cols = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual' \
                    ,'FireplaceQu','GarageQual','GarageCond','BsmtExposure','BsmtFinType1' \
                   ,'BsmtFinType2', 'Functional','GarageFinish','Fence','Street','Alley','LandSlope'\
                    ,'PavedDrive' ,'CentralAir','PoolQC','OverallQual','OverallCond','LotShape']

# Get list of categorical variables in holiday dataset
s = (df.dtypes == 'object')
object_cols = list(s[s].index)
object_cols = [x for x in object_cols if x not in modified_cols]

full = pd.merge(left = df, right = y , left_index= True, right_index = True)
#full['SalePrice'] = np.exp(full['SalePrice']) #THIS GOES BONKERS - must have been previously logged!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

''' #MANY data examination plots
for col in object_cols:
    if full[col].nunique()> 1:
        display(full.groupby(col)['SalePrice'].describe())
        print("\nSummary statistics and graph for "+ col)
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
        sns.countplot(data = full, x=col, ax= ax[0])
        ax[0].title.set_text("Count plot of " + col)
        sns.swarmplot(data=full,x=col,y='SalePrice', ax= ax[1])
        ax[1].title.set_text("Swarm plot of " + col +" versus Sale Price")
        if (full[col].nunique()>=15):
            ax[0].tick_params('x',labelrotation=70)
            ax[1].tick_params('x',labelrotation=70)
        fig.tight_layout()
        plt.show()
'''


#Feature Encoding Round 2 (Simplification + Ordinal): https://www.kaggle.com/code/limyenwee?scriptVersionId=90403458&cellId=76
cond_1_keep = ['Norm','Feedr','Artery']
roof_style_keep = ['Gable','Hip']
foundation_keep = ['PConc','CBlock','BrkTil']
garage_keep = ['Attchd','Detchd','BuiltIn']
sale_keep = ['WD','New','COD']
sale_cond_keep = ['Normal','Abnorml','Partial']
peak_months = ['5','6','7']
lot_config_keep = ['Inside','Corner','CulDSac']
unfinished_style = ['1.5Unf','2.5Unf']
exter_remove = ['AsphShn','BrkComm','CBlock','ImStucc','Stone']
for table in [df,test]:
    table.loc[table['LandContour']!='Lvl','LandContour'] = 0
    table.loc[table['LandContour']!=0,'LandContour'] = 1
    
    table.loc[~table['Condition1'].isin(cond_1_keep),'Condition1'] = "Others"
    table.loc[table['Condition2']!="Norm",'Condition2'] = 0
    table.loc[table['Condition2']!= 0,'Condition2'] = 1
    
    table.loc[~table['RoofStyle'].isin(roof_style_keep),'RoofStyle'] = "Others"
    table.loc[table['RoofMatl']!='CompShg','RoofMatl'] = 0
    table.loc[table['RoofMatl']!=0,'RoofMatl'] = 1
    
    table.loc[~table['Foundation'].isin(foundation_keep),'Foundation'] = "Others"
    table.loc[table['Heating']!='GasA','Heating'] = 0
    table.loc[table['Heating']=='GasA','Heating'] = 1
    table.loc[table['Electrical']!='SBrkr','Electrical'] = 0
    table.loc[table['Electrical']!=0,'Electrical'] = 1
    
    table.loc[~table['GarageType'].isin(garage_keep),'GarageType'] = "Others"
    
    table.loc[~table['SaleType'].isin(sale_keep),'SaleType'] = "Others"
    table.loc[~table['SaleCondition'].isin(sale_cond_keep),'SaleCondition'] = "Others"
    table.loc[~table['SaleCondition'].isin(sale_cond_keep),'SaleCondition'] = "Others"
    
    table.loc[table['Exterior1st'].isin(exter_remove),'Exterior1st'] = "Others"
    table.loc[table['Exterior2nd'].isin(exter_remove),'Exterior2nd'] = "Others"
    
    table.loc[table['MoSold'].isin(peak_months),'PeakMonths'] = 1
    table.loc[table['PeakMonths']!=1,'PeakMonths'] = 0
    
    table.loc[~table['LotConfig'].isin(lot_config_keep),'LotConfig'] = "Others"
    
    table.loc[~table['HouseStyle'].isin(unfinished_style),'Unfinished'] = 1
    table.loc[table['Unfinished']!= 1 ,'Unfinished'] = 0
    table.loc[table['HouseStyle'].isin(['SFoyer','SLvl']),'IsSplit'] = 1
    table.loc[table['IsSplit']!= 1 ,'IsSplit'] = 0   
    table["HouseStyle"] = table["HouseStyle"].map({'SFoyer': 0, "SLvl": 0, "1Story": 1, "1.5Fin": 2, "1.5Unf": 2, "2Story": 3, "2.5Fin": 4, "2.5Unf": 4})
    
    table.drop('Utilities', axis = 1 , inplace = True)

    
modified_cols_round_2 = ['HouseStyle','LandContour','Condition2','RoofMatl','Heating','Electrical','Utilities']
object_cols = [x for x in object_cols if x not in modified_cols_round_2]


#some of the features still have a lot of unique values.
#usng Kmeans
neighborhood = full.groupby(['Neighborhood'])['SalePrice'].describe()
neighborhood_cluster = KMeans(n_clusters=5, random_state = 927)
neighborhood_cluster.fit(neighborhood)

neigh_cluster_table = pd.DataFrame(zip(list(neighborhood.index),list(neighborhood.loc[:,'mean']),list(neighborhood_cluster.labels_)),columns = ['Neighborhood','MeanSalePrice','Neighborhood Cluster'])
for i  in range(len(neigh_cluster_table.groupby('Neighborhood Cluster')['Neighborhood'].unique())):
    print("Cluster " + str(i))
    print(neigh_cluster_table.groupby('Neighborhood Cluster')['Neighborhood'].unique()[i])
sns.scatterplot(data = neigh_cluster_table, x='Neighborhood',y = 'MeanSalePrice', hue='Neighborhood Cluster',palette=sns.color_palette("Set2",5))
plt.xticks(rotation=70)
plt.show()

subclass = full.groupby(['MSSubClass'])['SalePrice'].describe()

subclass_cluster = KMeans(n_clusters=4, random_state = 927)
subclass_cluster.fit(subclass)

mssub_cluster_table = pd.DataFrame(zip(list(subclass.index),list(subclass.loc[:,'mean']),list(subclass_cluster.labels_)),columns = ['MSSubClass','MeanSalePrice','MSSubClass Cluster'])
for i  in range(len(mssub_cluster_table.groupby('MSSubClass Cluster')['MSSubClass'].unique())):
    print("Cluster " + str(i))
    print(mssub_cluster_table.groupby('MSSubClass Cluster')['MSSubClass'].unique()[i])
sns.scatterplot(data = mssub_cluster_table, x='MSSubClass',y = 'MeanSalePrice', hue='MSSubClass Cluster',palette=sns.color_palette("Set2",4))
plt.xticks(rotation=70)
plt.show()

mssub_cluster_table.drop('MeanSalePrice', axis = 1 ,inplace = True)
neigh_cluster_table.drop('MeanSalePrice', axis = 1, inplace = True)

df = pd.merge(left = df.reset_index(), right = mssub_cluster_table, how='left', on ='MSSubClass').set_index('Id')
df = pd.merge(left = df.reset_index(), right = neigh_cluster_table, how='left', on ='Neighborhood').set_index('Id')
df.drop('MSSubClass', axis = 1 ,inplace = True)
df.drop('Neighborhood', axis = 1 ,inplace = True)

test = pd.merge(left = test.reset_index(), right = mssub_cluster_table, how='left', on ='MSSubClass').set_index('Id')
test = pd.merge(left = test.reset_index(), right = neigh_cluster_table, how='left', on ='Neighborhood').set_index('Id')
test.drop('MSSubClass', axis = 1 ,inplace = True)
test.drop('Neighborhood', axis = 1 ,inplace = True)

modified_cols.append('MSSubClass')
modified_cols.append('Neighborhood')

object_cols.append('MSSubClass Cluster')
object_cols.append('Neighborhood Cluster')
object_cols.remove('MSSubClass')
object_cols.remove('Neighborhood')

#We perform one-hot encoding to the remaining categorical variables
# One Hot Encoding for Other Columns
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(df[object_cols]))
OH_cols.index = df.index
OH_cols.columns = OH_encoder.get_feature_names(object_cols)
df = df.drop(object_cols, axis=1)
df = pd.concat([df, OH_cols], axis=1)

OH_cols = pd.DataFrame(OH_encoder.transform(test[object_cols]))
OH_cols.index = test.index
OH_cols.columns = OH_encoder.get_feature_names(object_cols)
test = test.drop(object_cols, axis=1)
test = pd.concat([test, OH_cols], axis=1)

#ake care of the skewness of the features in our dataset. We use skew() from the scipy.stats module to identify which columns are skewed.
#Any skewness greater than 0.5 is actually considered slightly skewed hence we will perform log-transformation for any values greather than that.
skewed = df[df.columns[~df.columns.isin(list(OH_cols.columns) + modified_cols + object_cols)]].apply(lambda x: skew(x.dropna().astype(float)))
skewed = skewed[skewed > 0.5]
skewed = skewed.index

df[skewed] = np.log1p(df[skewed])
test[skewed] = np.log1p(test[skewed])

#While log-transformation took care of the skewness in the features, we will also want to further scale the features to a standardize the range.
for col in df[df.columns]:
    if col not in (list(OH_cols.columns) + modified_cols + object_cols):
        scaler = RobustScaler()
        df[col] = scaler.fit_transform(df[[col]])
        test[col] = scaler.transform(test[[col]])


full = pd.merge(left = df, right = y , left_index= True, right_index = True)
mi = mutual_info_regression(X = full.drop('SalePrice', axis = 1), y = full['SalePrice'])
mi_df = pd.DataFrame(list(zip(full.columns,mi)), columns =['Feature','Mutual Info'])
mi_df = mi_df.sort_values('Mutual Info',ascending=False)

low_mi_df = mi_df[abs(mi_df['Mutual Info']) == 0]
filter_feature = sorted(list(low_mi_df['Feature']))
print("Number of low correlated features dropped: " + str(len(filter_feature)))
df = df.drop(filter_feature,axis=1)
test = test.drop(filter_feature,axis=1)


#create new polynomial and interaction features from the high mutual information features to derive new combinations that might be useful to our model later on
#read more about interaction features: https://stattrek.com/multiple-regression/interaction.aspx
top_mi_list = list(mi_df.head(20)['Feature'])
top_mi_subset = df[top_mi_list]
index_copy = top_mi_subset.index

poly = PolynomialFeatures(2, interaction_only=True)
poly_features = pd.DataFrame(poly.fit_transform(top_mi_subset),columns=poly.get_feature_names_out(top_mi_list))
poly_features = poly_features.iloc[:,len(top_mi_list) + 1:]
poly_features.set_index(index_copy, inplace = True)
poly_and_price = pd.concat([y,poly_features],axis=1).dropna()
top_20_poly = abs(poly_and_price.corr()['SalePrice']).sort_values(ascending=False)[1:21]

df = pd.concat([df,poly_features[top_20_poly.index]],axis=1)

top_mi_subset = test[top_mi_list]
index_copy = top_mi_subset.index
poly_features = pd.DataFrame(poly.transform(top_mi_subset),columns=poly.get_feature_names_out(top_mi_list))
poly_features = poly_features.iloc[:,len(top_mi_list) + 1:]
poly_features.set_index(index_copy, inplace = True)
test = pd.concat([test,poly_features[top_20_poly.index]],axis=1)

#Outlier id by Isolation Forests
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(random_state=0)
df_without_outlier = pd.Series(iso_forest.fit_predict(df), index = full.index)
df = df.loc[df_without_outlier.index[df_without_outlier == 1],:]

full = pd.merge(left = df, right = y , left_index= True, right_index = True)
linear = LinearRegression()
Y = full['SalePrice']
linear.fit(full.drop(['SalePrice'],axis=1), Y)
Y_hat = linear.predict(full.drop(['SalePrice'],axis=1))
residuals = Y - Y_hat
y_vs_yhat_df = pd.DataFrame(zip(Y.values,Y_hat,residuals),columns=['y','yhat','residuals'],index=full.index)

r2 = r2_score(Y, Y_hat)
print("About " + str(round(r2 * 100,2)) + "% of variation in the Sale Price can be explained by the model.")

sns.scatterplot(Y, Y_hat)
sns.lineplot(np.linspace(10.5,13.5),np.linspace(10.5,13.5), color='black', linewidth=2.5)
plt.show()

standard_residuals = (residuals - residuals.mean()) / residuals.std()
outliers = full[abs(standard_residuals) > 3]
y_vs_yhat_df.loc[y_vs_yhat_df.index.isin(outliers.index),'Outlier'] = 1
y_vs_yhat_df.loc[y_vs_yhat_df['Outlier'] != 1 ,'Outlier'] = 0

sns.scatterplot(data = y_vs_yhat_df, x='y', y='yhat',hue ='Outlier', palette = ['blue','red'])
sns.lineplot(np.linspace(10.5,13.5),np.linspace(10.5,13.5), color='black', linewidth=2.5)
plt.show()

df = df.loc[y_vs_yhat_df[y_vs_yhat_df['Outlier'] == 0].index, :]

df = df.drop(list(test.columns[test.nunique()== 1 ]), axis=1)
test = test.drop(list(test.columns[test.nunique()== 1]), axis=1)



#MODELLING
full = pd.merge(left = df, right = y , left_index= True, right_index = True)
train_y = np.log(full['SalePrice'])
train_X = full.drop(['SalePrice'], axis=1)


dev_train, dev_test = train_test_split(full, test_size=0.2, shuffle=True)
dev_train_y = dev_train['SalePrice']
dev_train_X = dev_train.drop(['SalePrice'], axis=1)
dev_test_y = dev_test['SalePrice']
dev_test_X = dev_test.drop(['SalePrice'], axis=1)


#NOW MODELLING IS NOT FOLLOWING Lim
#TRAINING and EVALUATING on the Training Set p112 (REGRESSION)

#Look at X-val scores
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

#MODEL ACCURACY EVALUATOR
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(train_X, train_y)

#mean_squared_error()
lin_predictions = lin_reg.predict(train_X)
lin_mse = mean_squared_error(train_y, lin_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse) #0.0844


#Better to Evaluate Using Cross-Validation p114
#Scikit-Learn’s K-fold cross-validation feature
#randomly splits the training set into 10 distinct subsets
lin_scores = cross_val_score(lin_reg, train_X, train_y, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores) #mu = 0.0982; sd = 0.0067

#eval lin_reg 
evaluate(lin_reg, train_X, train_y) #Accuracy = 99.47%.

#MAKE KAGGLE SUBMISSION
predictions = lin_reg.predict(test)
df = pd.DataFrame(zip(test.index, np.exp(predictions)))
df.columns = ['Id', 'SalePrice']
df.to_csv('MyPreds_lin_reg_LimEng.csv', index=False) #Score: 0.13431 uploaded on 14/04/2022



#let’s try a more complex model (nonlinear decision tree) to see how it does p113
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(train_X, train_y)

evaluate(tree_reg, train_X, train_y) #error: 0; accuracy: 100%

#Eval dec tree with cross val to compare scores
tree_scores = cross_val_score(tree_reg, train_X, train_y, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)
display_scores(tree_rmse_scores) ##mu = 0.173; sd = 0.013


#Randon Forest p115
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(train_X, train_y)

rf_predictions = forest_reg.predict(train_X)
forest_mse = mean_squared_error(train_y, rf_predictions)
forest_rmse = np.sqrt(forest_mse) #approx 7713 - much better!
print(forest_rmse) #0.0445 better cf lin_reg

#x-val
rf_scores = cross_val_score(forest_reg, train_X, train_y, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-rf_scores)
display_scores(forest_rmse_scores) #mu = 0.118; sd = 0.0092 - lin_reg better! 

evaluate(forest_reg, train_X, train_y) #Accuracy = 99.73%. - better than lin_reg

joblib.dump(forest_reg, "forest_reg_YenWeeLim.pkl")
#TO LOAD: forest_reg_loaded = joblib.load("forest_reg_YenWeeLim.pkl")

predictions = forest_reg.predict(test)
df = pd.DataFrame(zip(test.index, np.exp(predictions)))
df.columns = ['Id', 'SalePrice']
df.to_csv('MyPreds_forest_reg_LimEng.csv', index=False) #Score: 0.14593 uploaded on 14/04/2022

#SUMMARY
#Lin_reg better on x-val and from Kaggle scores

#NOW AGAIN FOLLOW YEN WEE LIM: 
#Stacked regressor is a type of Level 1 ensemble model that generalizes the predictions made by different models to get the final output. You can study more information regarding stacked models here
#https://www.analyticsvidhya.com/blog/2020/12/improve-predictive-model-score-stacking-regressor/
#MORE stuff on Stacked Regressions: https://www.kaggle.com/code/serigne/stacked-regressions-top-4-on-leaderboard/notebook

#NB I use logs here - must relate to use of np.exp() above
full = pd.merge(left = df, right = y , left_index= True, right_index = True)
full['SalePrice'] = np.log(full['SalePrice'])
train_y = full['SalePrice']
train_X = full.drop(['SalePrice'], axis=1)

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


start = time.time() #takes 9mins
print("Recording Modelling Time")
for i in [ridgemodel, hubermodel, cbmodel, svrmodel, xgbmodel, stackmodel]:
    i.fit(train_X,train_y)
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
fit = (svrmodel.predict(train_X) + xgbmodel.predict(train_X) +   stackmodel.predict(train_X) + ridgemodel.predict(train_X) + hubermodel.predict(train_X) + cbmodel.predict(train_X)) / 6
print(mean_squared_error(train_y, fit, squared=False))



#Next fit the models with all the data.
start = time.time()
print("Recording Modelling Time")
for i in [ridgemodel, hubermodel, cbmodel, svrmodel, xgbmodel, stackmodel]:
    i.fit(train_X, train_y)
    if i == stackmodel:
        i.fit(np.array(train_X), np.array(train_y))
end = time.time()
print("Time Elapsed: " + str(round((end - start)/60,0)) +"minutes.")


final_prediction = (np.exp(ridgemodel.predict(test)) + 3 * np.exp(xgbmodel.predict(test)) \
+  5 * np.exp(stackmodel.predict(test)) + 4 * np.exp(svrmodel.predict(test)) \
+  np.exp(hubermodel.predict(test)) +  np.exp(cbmodel.predict(test))) / 15

submission = pd.DataFrame(final_prediction, index = test.index)

submission.reset_index(drop=False, inplace = True)
submission = submission.rename(columns={0 : 'SalePrice', 'index' : 'Id'})
submission.to_csv('Preds_followingLim.csv', index=False) # Score 0.12286 on 13/04/2022


#USE RandomizedSearchCV, from https://github.com/ageron/handson-ml2/blob/master/02_end_to_end_machine_learning_project.ipynb
from sklearn.model_selection import RandomizedSearchCV
#view settings
for key in RandomForestRegressor().get_params().keys():
    print(key, RandomForestRegressor().get_params().get(key))

#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
#docs tell us most important settings:
#number of trees in the forest (n_estimators), and
#no. features considered for splitting at each leaf node (max_features)
#following: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
#also try:
#max_depth = max number of levels in each decision tree
#min_samples_split = min number of data points placed in a node before the node is split
#min_samples_leaf = min number of data points allowed in a leaf node
#bootstrap = method for sampling data points (with or without replacement)
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}



forest_reg_rndSrch = RandomForestRegressor()
rnd_search = RandomizedSearchCV(forest_reg_rndSrch, param_distributions=random_grid,
                                n_iter=500, cv=8, scoring='neg_mean_squared_error',
                                verbose=2, random_state=42, n_jobs = 2)

#RUN rndm search 5-6hrs on ASUS 
rnd_search.fit(train_X, train_y)
best_random = rnd_search.best_estimator_
joblib.dump(best_random, "LimEng_RandomizedSearchCV_best_random.pkl") #save
#rnd_search.best_params_

#load the model - data has 322 feats after removing four erroneous feats
best_random_loaded = joblib.load("LimEng_RandomizedSearchCV_best_random.pkl")

evaluate(best_random_loaded, train_X, train_y) #Accuracy = 99.73%. - better than lin_reg

#x-val
rf_scores = cross_val_score(best_random_loaded, train_X, train_y, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-rf_scores)
display_scores(forest_rmse_scores) #mu = 0.122; sd = 0.0121 - lin_reg better! 

#MAKE KAGGLE SUBMISSION
predictions = best_random_loaded.predict(test)
df = pd.DataFrame(zip(test.index, np.exp(predictions)))
df.columns = ['Id', 'SalePrice']
df.to_csv('RandomForestRegressor_bestMod.csv', index=False) #uploaded on 11/04/2022 Score 0.4!!!!!!!!!!!!!!!!!


