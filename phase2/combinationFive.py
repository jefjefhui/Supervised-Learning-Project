import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB



#Read the CSV file by panda
dataset = pd.read_csv("Ecoli.csv")

#Select the target/features area as X
X = dataset.loc[:, '(Num) Column 1 ':'(Nom) Column 116']

#Select the label as y
y = dataset.loc[:, 'Target (Column 117)']

#Display settings to show 116 columns of data and the fist and last 100 rows of data
pd.set_option('display.width',2000)
pd.set_option('display.max_columns',116)
pd.set_option('display.max_rows', 200)
pd.set_option('display.min_rows',None)


#This check how many null values in the dataset
#print(dataset.isnull().sum())
#print(X.isnull().sum())
#print(y.isnull().sum())

#print(y)

#Implement combination 5 (Isolation forest, Standardization, KNNImputer)
#1 Implement KNN Imputer
knnI = KNNImputer()
KnnIAfter = knnI.fit_transform(X)
#print(X.isnull().sum())
#zzz = pd.DataFrame(KnnIAfter)
#print(zzz.isnull().sum())


#2 Implement Standardization
SAD = StandardScaler()
SADAfter = SAD.fit_transform(KnnIAfter)

#ddd = X.loc[:,'(Num) Column 6']
#print(ddd.max())
#print(ddd.min())

#eee =pd.DataFrame(SADAfter)
#fff = eee.loc[:,5]
#print(fff.shape)
#print(fff.max())
#print(fff.min())


#ggg = X.loc[:,'(Num) Column 3']
#print(ggg.max())
#print(ggg.min())

#hhh = eee.loc[:,2]
#print(hhh.max())
#print(hhh.min())

ao = pd.DataFrame(SADAfter)
#3 Isolation forest
IF = IsolationForest(random_state=0)


#split the data to 2sets
#set 1
set1Training = ao.loc[0:749,:]
print(set1Training.shape)
set1Testing = ao.loc[750:1499,:]
print(set1Testing.shape)
test1 = IF.fit(set1Training)
test1Result = test1.predict(set1Testing)
print(test1Result)
rr=0
for x in test1Result:
    if x==-1:
        rr+=1
print(rr)

#set2
set2Training = ao.loc[750:1499,:]
print(set2Training.shape)
set2Testing = ao.loc[0:749,:]
print(set2Testing.shape)
test2 = IF.fit(set2Training)
test2Result = test2.predict(set2Testing)
print(test2Result)
ww = 0
for x in test2Result:
    if x==-1:
        ww+=1
print(ww)


#location the outlier position
#290,294,296,298,301,305,311,313,318,322,323,328,329

print(test2Result[289])
print(test2Result[293])
print(test2Result[295])
print(test2Result[297])
print(test2Result[300])
print(test2Result[304])
print(test2Result[310])
print(test2Result[312])
print(test2Result[317])
print(test2Result[321])
print(test2Result[322])
print(test2Result[327])
print(test2Result[328])


#Remove all the outerliers
#Remove outliers for X

finalProcessedX = ao.drop(ao.index[[289,293,295,297,300,304,310,312,317,321,322,327,328]],
                          axis=0)

print(finalProcessedX.shape)


#Remove outliers for y
convertytodf = pd.DataFrame(y)
finalProcessedy = convertytodf.drop(convertytodf.index[[289,293,295,297,300,304,310,312,317,321,322,327,328]],
                                    axis=0)
print(finalProcessedy.shape)





#Implement Decision tree
dtc = DecisionTreeClassifier(random_state=0)
dtcResult = cross_val_score(dtc,finalProcessedX,finalProcessedy,cv=10,scoring='f1')
print(dtcResult.mean())




#Implment random forest
rfc = RandomForestClassifier(random_state=0)
rfcResult = cross_val_score(rfc,finalProcessedX,finalProcessedy.values.ravel(),cv=10,scoring='f1')
print(rfcResult.mean())



#Implment KNN
knnc = KNeighborsClassifier()
knncResult = cross_val_score(knnc,finalProcessedX,finalProcessedy.values.ravel(),cv=10,scoring='f1')
print(knncResult.mean())



#Implment Naive Bayes
gauNB = GaussianNB()
gauNBResult =cross_val_score(gauNB,finalProcessedX,finalProcessedy.values.ravel(),cv=10,scoring='f1')
print(gauNBResult.mean())





#Calculate the average score
sum = dtcResult.mean()+rfcResult.mean()+knncResult.mean()+gauNBResult.mean()
average = sum/4

print("The average score of combination 5 is "+ str(average))


# 1st attempt: 0.745   2nd attempt: 0.745  3rd attempt:0.741  final average: 0.744