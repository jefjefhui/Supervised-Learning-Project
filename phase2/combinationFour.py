import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
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


#Implement combination 4 (Isolation Forest, Standardization,SimpleImputer)
#1 Implement SimpleImputer
SI =SimpleImputer(missing_values=np.nan,strategy='mean')
SIAfter = SI.fit_transform(X)
#rr = pd.DataFrame(SIAfter)
#print(rr.isnull().sum())


#2 Implement standardization
SS = StandardScaler()
SSResult = SS.fit_transform(SIAfter)

#checkstand = X.loc[:,'(Num) Column 6']
#print(checkstand.max())
#print(checkstand.min())

#eee = pd.DataFrame(SSResult)
#fff = eee.loc[:,5]
#print(fff.max())
#print(fff.min())

#checkstand2 = X.loc[:,'(Num) Column 3']
#print(checkstand2.max())
#print(checkstand2.min())
#ggg= eee.loc[:,2]
#print(ggg.max())
#print(ggg.min())


#3 Implement isolation forest
IF = IsolationForest(random_state=0)

#print(type(SSResult))
et = pd.DataFrame(SSResult)
#print(type(et))


#split into 2 sets of data
set1Training = et.loc[0:749,:]
set1Testing = et.loc[750:1499,:]
testing1 = IF.fit(set1Training)
testing1_Result = testing1.predict(set1Testing)
print(testing1_Result)
aj =0
for x in testing1_Result:
    if x==-1:
        aj+=1
print(aj)
#print(set1Training.shape)
#print(set1Testing.shape)

set2Training = et.loc[750:1499,:]
set2Testing =et.loc[0:749,:]
#print(set2Training.shape)
#print(set2Testing.shape)
testing2 = IF.fit(set2Training)
testing2Result = testing2.predict(set2Testing)
print(testing2Result)
oo = 0
for x in testing2Result:
    if x==-1:
        oo+=1
print(oo)


#Retrive the outlier location
#296,298,301,305,311,318,322,323,328,329
print(testing2Result[295])
print(testing2Result[297])
print(testing2Result[300])
print(testing2Result[304])
print(testing2Result[310])
print(testing2Result[317])
print(testing2Result[321])
print(testing2Result[322])
print(testing2Result[327])
print(testing2Result[328])


#Remove outliers from x
finalProcessedX = et.drop(et.index[[295,297,300,304,310,317,321,322,327,328]],
                          axis=0)

print(finalProcessedX.shape)


#Remove outliers from y
#print(type(y))
#print(y.shape)
changeytoDF = pd.DataFrame(y)
finalProcessedy = changeytoDF.drop(changeytoDF.index[[295,297,300,304,310,317,321,322,327,328]],
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



#Implement Naive Bayes
gauNB = GaussianNB()
gauNBResult = cross_val_score(gauNB,finalProcessedX,finalProcessedy.values.ravel(), cv=10,scoring='f1')
print(gauNBResult.mean())



#Calculate the average score for combination 4
sum = dtcResult.mean()+ rfcResult.mean()+knncResult.mean()+gauNBResult.mean()
average = sum/4

print('The average score of combination 4 is  '+ str(average))


#first attempt: 0.742 second attempt:0.742  third attempt:0.742  final average: 0.742