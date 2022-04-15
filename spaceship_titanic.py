import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

train = pd.read_csv('train.csv', header=0)
test = pd.read_csv('test.csv', header=0)
space = pd.concat([train, test])

X = space.iloc[:, :-1]
y = train.iloc[:, -1]

X.Name = X.Name.fillna('no name')
X['surname'] = X.iloc[:, -1:].applymap(lambda x: x.split(' ')[1])

names = X.surname.tolist()
X['family'] = X.iloc[:, -1:].applymap(lambda x: names.count(x))
X.family[X.family ==  1] = 1
X.family[X.family >  1] = 2
X.family[X.surname == 'name'] = 0

del X['Name']

X.Cabin = X.Cabin.fillna('u/u/u')
X['deck'] = X.iloc[:, 3:4].applymap(lambda x: x.split('/')[0])
X['side'] = X.iloc[:, 3:4].applymap(lambda x: x.split('/')[2])

del X['Cabin']

#impute missing vals for numerical
#COULD USE REGRESSION
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
cols = X.columns[6:11].tolist() + ['Age']
imputer.fit(X[cols])
I = imputer.transform(X[cols])
new = pd.DataFrame(I, columns=cols, index=X.index)
X[cols] = new[cols]

X = X.fillna('unk')

#hot encode
from sklearn.preprocessing import OneHotEncoder

to_encode = ['HomePlanet','CryoSleep','Destination','VIP', 'family','deck','side']
X = X.reset_index(drop=True)

X.CryoSleep[X.CryoSleep == True] = 'sleep'
X.CryoSleep[X.CryoSleep == False] = 'awake'

X.VIP[X.VIP == True] = 'vip'
X.VIP[X.VIP == False] = 'norm'

for enc in to_encode:
    cat_encoder = OneHotEncoder()
    X_cat_1hot = cat_encoder.fit_transform(X[[enc]])

    X = X.drop(enc, axis=1)

    X = pd.concat([X, pd.DataFrame(X_cat_1hot.toarray())], axis=1)
    X.columns = list(X.columns[:(X_cat_1hot.shape[1] * -1)]) + [f"{enc}_{qw}" for qw in range(X_cat_1hot.shape[1])]


del X['PassengerId']
del X['surname']

train_x = X.iloc[:train.shape[0], :]
test_x = X.iloc[train.shape[0]:, :]

sys.exit(2)

#p.134 Stochastic Gradient Descent (SGD) classifier
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42) #reproducible param
sgd_clf.fit(train_x, y)

predictions = sgd_clf.predict(test_x)
df = pd.DataFrame(zip(test.PassengerId, predictions))
df.columns = ['PassengerId', 'Transported']
df.to_csv('MyPreds_SGD.csv', index=False) #uploaded on 11/04/2022



#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(train_x, y)

predictions = forest_clf.predict(test_x)
df = pd.DataFrame(zip(test.PassengerId, predictions))
df.columns = ['PassengerId', 'Transported']
df.to_csv('MyPreds_rfClass.csv', index=False) #uploaded on 11/04/2022


from sklearn.neighbors import KNeighborsClassifier
knb = KNeighborsClassifier()
knb.fit(train_x, y)

predictions = knb.predict(test_x)
df = pd.DataFrame(zip(test.PassengerId, predictions))
df.columns = ['PassengerId', 'Transported']
df.to_csv('MyPreds_K_neighborsClass.csv', index=False) #uploaded on 11/04/2022


#SUMMARY
#rf gets 79% accuracy (highest is 81%) - Concentrate on Ames Housing!!

#p136 Evaluating a classifier is often significantly trickier than evaluating a regressor
#The following code does roughly the same thing as Scikit-Learn’s cross_val_score() function, and it prints the same result:
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) 
for train_index, test_index in skfolds.split(train_x, y):
    clone_clf = clone(sgd_clf)
    X_train_folds = train_x.loc[train_index]
    y_train_folds = y.loc[train_index]
    X_test_fold = train_x.loc[test_index]
    y_test_fold = y.loc[test_index]
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred)) # rf: .80, .79, .78; sgd: .77, .77, .75

#Xval p137
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, train_x, y, cv=3, scoring="accuracy")

from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

#Can you guess this model’s accuracy? Let’s find out:
never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, train_x, y, cv=3, scoring="accuracy")

#50% accuracy!! remember 50% of data are 'True'!! p137

#Confusion Matrix (CM) p137
#count the number of times instances of class A are classified as class B. For example, to know the
#number of times the classifier confused images of 5s with 3s, you would look in the fifth row and third column of the confusion matrix

#CM reqs predictions - this just for 5s
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, train_x, y, cv=3)

from sklearn.metrics import confusion_matrix
confusion_matrix(y, y_train_pred) #rf: 78%; sgd .71

#see p140 for schematic of CM
#calc precision and recall
from sklearn.metrics import precision_score, recall_score
precision_score(y, y_train_pred) 
recall_score(y, y_train_pred)  #When it claims an image represents a 5, it is correct only ca80% of the time. Moreover, it only detects 65% of the 5s (numbers may vary - prob coming from random state issue on p136)

#calc F1 score (precision and recall combined) - favors classifiers that have similar precision and recall
from sklearn.metrics import f1_score
f1_score(y, y_train_pred)

# can alter prediction outcomes by altering threshold
y_scores = sgd_clf.decision_function([train_x.iloc[0]])
y_scores
threshold = 0 #default value
y_some_digit_pred = (y_scores > threshold)

threshold = 8000 #altered val
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred #now False - raising the threshold decreases recall

#decide thershold with the cross_val_predict() function
y_scores = cross_val_predict(sgd_clf, train_x, y, cv=3, method="decision_function")

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y, y_scores)

#plot precision and recall as functions of the threshold value p143
#PR curve
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    #[...] # highlight the threshold and add the legend, axis label, and grid
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    plt.show()
    
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
#can decide what value to choose depending on your reqs for either (high) precision or (high) recall
#some code issue here but works depite error msg
#see p141 for when you need good precision vs good recall
#could also plot precision v recall p145

#calc thresh for min 90% precision
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)] #NB my thresh is way - see earlier
#use it:
y_train_pred_90 = (y_scores >= threshold_90_precision)

precision_score(y, y_train_pred_90) #although my performance is almost same
recall_score(y, y_train_pred_90)

#p146 ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal
    [...] # Add axis labels and grid
    plot_roc_curve(fpr, tpr)
    plt.show()
    
#calc AUC
from sklearn.metrics import roc_auc_score
roc_auc_score(y, y_scores)


#Try SVC
from sklearn.svm import SVC
svm_clf = SVC(gamma="auto")
svm_clf.fit(train_x, y)

from sklearn.model_selection import cross_val_score
svm_scores = cross_val_score(svm_clf, train_x, y, cv=10)
svm_scores.mean()


skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) 
for train_index, test_index in skfolds.split(train_x, y):
    clone_clf = clone(svm_clf)
    X_train_folds = train_x.loc[train_index]
    y_train_folds = y.loc[train_index]
    X_test_fold = train_x.loc[test_index]
    y_test_fold = y.loc[test_index]
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred)) # rf: .80, .79, .78; sgd: .77, .77, .75

predictions = svm_clf.predict(test_x)
df = pd.DataFrame(zip(test.PassengerId, predictions))
df.columns = ['PassengerId', 'Transported']
df.to_csv('MyPreds_SVC.csv', index=False) #uploaded on 11/04/2022


# convert age to buckets as young people survive
train_x["AgeBucket"] = train_x["Age"] // 15 * 15
test_x["AgeBucket"] = test_x["Age"] // 15 * 15


#try with scale training & test data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_x)
X = scaler.transform(train_x)
train_x = pd.DataFrame(X, columns=train_x.columns, index=train_x.index)

scaler.fit(test_x)
X = scaler.transform(test_x)
test_x = pd.DataFrame(X, columns=test_x.columns, index=test_x.index)

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
forest_clf_scl = RandomForestClassifier(random_state=42)
forest_clf_scl.fit(train_x, y)

skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) 
for train_index, test_index in skfolds.split(train_x, y):
    clone_clf = clone(forest_clf_scl)
    X_train_folds = train_x.loc[train_index]
    y_train_folds = y.loc[train_index]
    X_test_fold = train_x.loc[test_index]
    y_test_fold = y.loc[test_index]
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred)) #.80, .79, .78


predictions = forest_clf_scl.predict(test_x)
df = pd.DataFrame(zip(test.PassengerId, predictions))
df.columns = ['PassengerId', 'Transported']
df.to_csv('MyPreds_rfClass_scaledAgeBucket.csv', index=False) # Score 0.77 uploaded on 11/04/2022 worse than unscaled

#Could do random grid search but see Ames_housing for example


#Graphically compare performance btwn models
from sklearn.model_selection import cross_val_score
forest_clf100 = RandomForestClassifier(n_estimators=100, random_state=42)
forest_scores100 = cross_val_score(forest_clf100, train_x, y, cv=10)
forest_scores = cross_val_score(forest_clf, train_x, y, cv=10)
forest_scores.mean()

#compare cvals (10 each) for diff mods w/wo scaling etc
plt.figure(figsize=(8, 4))
plt.plot([1]*10, forest_scores100, ".")
plt.plot([2]*10, forest_scores, ".")
plt.boxplot([forest_scores100, forest_scores], labels=("RF100","Random Forest"))
plt.ylabel("Accuracy", fontsize=14)
plt.show()






