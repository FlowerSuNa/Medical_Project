
## 2. Preporcessing Missing Values
## Import library and data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

medical = pd.read_csv("medical_dataset_outliers.csv")


## Check data
pd.set_option('display.max_columns', 100)

print(medical.columns)
print(medical.describe())


## Copy data
md = medical.copy()


## Check correlation
print(md.corr(method='pearson'))


## Remove missing values : Drinking status, Smoking status
print("Drinking status missing : %d" % len(md.loc[md['DRK_YN'].isnull()]))
print("Smoking status missing : %d" % len(md.loc[md['SMK_STAT_TYPE_CD'].isnull()]))

md = md.loc[md['DRK_YN'].isnull() != True]
md = md.loc[md['SMK_STAT_TYPE_CD'].isnull() != True]

print("Drinking status missing after removal : %d" % len(md.loc[md['DRK_YN'].isnull()]))
print("Smoking status missing after removal : %d" % len(md.loc[md['SMK_STAT_TYPE_CD'].isnull()]))


## Fill missing values : Eyesight
sl = len(medical.loc[medical['SIGHT_LEFT'].isnull()])
sr = len(medical.loc[medical['SIGHT_RIGHT'].isnull()])
sb = len(medical.loc[(medical['SIGHT_LEFT'].isnull()) & (medical['SIGHT_RIGHT'].isnull())])

median = md.groupby(['AGE_GROUP','SEX'], as_index=False)[['SIGHT_LEFT','SIGHT_RIGHT']].median()
median

md['SIGHT_LEFT'].fillna(md.groupby(['AGE_GROUP','SEX'])['SIGHT_LEFT'].transform('median'), inplace=True)
md['SIGHT_RIGHT'].fillna(md.groupby(['AGE_GROUP','SEX'])['SIGHT_RIGHT'].transform('median'), inplace=True)

print("Eyesight missing after handling : %d" % len(md.loc[md[['SIGHT_LEFT','SIGHT_RIGHT']].isnull()]))


## Fill missing values : Hearing
hl = len(medical.loc[medical['HEAR_LEFT'] == 2])
hr = len(medical.loc[medical['HEAR_RIGHT'] == 2])
hb = len(medical.loc[(medical['HEAR_LEFT'] == 2) & (medical['HEAR_RIGHT'] == 2)])

print("Left : %d" % hl)
print("Right : %d" % hr)
print("Both : %d" % hb)

hl = len(medical.loc[medical['HEAR_LEFT'].isnull()])
hr = len(medical.loc[medical['HEAR_RIGHT'].isnull()])
hb = len(medical.loc[(medical['HEAR_LEFT'].isnull()) & (medical['HEAR_RIGHT'].isnull())])

print("Left : %d" % hl)
print("Right : %d" % hr)
print("Both : %d" % hb)

md['HEAR_LEFT'].fillna(1, inplace=True)
md['HEAR_RIGHT'].fillna(1, inplace=True)

print("Hearing missing : " + md[['HEAR_LEFT','HEAR_RIGHT']].isnull().sum())


## Fill missing values : protein in urine
## Check missing values
print(len(md.loc[md['OLIG_PROTE_CD'].isnull()]))

## Divid data
from sklearn.model_selection import train_test_split

predict = md.loc[md['OLIG_PROTE_CD'].isnull()]
train = md.loc[md['OLIG_PROTE_CD'].isnull() != True]

feat = ['WAIST','BP_STATE','BLDS','CREATININE','GAMMA_GTP']
X_train, X_test, y_train, y_test = train_test_split(train[feat], train['OLIG_PROTE_CD'], random_state=0)

print("X_train : " + str(X_train.shape))
print("X_test : " + str(X_test.shape))
print("y_train : " + str(y_train.shape))
print("y_test : " + str(y_test.shape))

## Use Decision Tree
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=15, random_state=0)
tree.fit(X_train, y_train)

## Evaluate a model
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,tree.predict(X_test[feat])))
print(tree.score(X_train,y_train))
print(tree.score(X_test,y_test))
print(tree.feature_importances_)

## Use Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=0)
forest.fit(X_train, y_train)

## Evaluate a model
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,tree.predict(X_test[feat])))
print(forest.score(X_train, y_train))
print(forest.score(X_test, y_test))
print(forest.feature_importances_)

## Use Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbrt = GradientBoostingClassifier(max_depth=5, random_state=0)
gbrt.fit(X_train, y_train)

## Evaluate a model
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,tree.predict(X_test[feat])))
print(gbrt.score(X_train, y_train))
print(gbrt.score(X_test, y_test))
print(gbrt.feature_importances_)

## Use K Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=15)
clf.fit(X_train, y_train)

print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))

## Evaluate a model
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,tree.predict(X_test[feat])))
print(gbrt.score(X_train, y_train))
print(gbrt.score(X_test, y_test))
print(gbrt.feature_importances_)

## Use Kernelized Support Vector Machines
from sklearn.svm import SVC

svc = SVC(C=10, gamma=10)
svc.fit(X_train, y_train)

## Evaluate a model
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,tree.predict(X_test[feat])))
print(gbrt.score(X_train, y_train))
print(gbrt.score(X_test, y_test))
print(gbrt.feature_importances_)

## Predict missing values
## md['OLIG_PROTE_CD'].loc[md['OLIG_PROTE_CD'].isnull()] = tree.predict(predict[feat])
## print(md['OLIG_PROTE_CD'].isnull().sum())



md.columns

## Save dataset
md.to_csv("medical_dataset_missing_values.csv", index=False)
md = pd.read_csv("medical_dataset_missing_values.csv")
md.info()