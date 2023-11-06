import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB


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


#Start implementing combination Two (Isolation Forest,Max-Min Normalization,KNNImputer)
#1 Implement KNNImputer
comboTwoKNNI = KNNImputer()
comboTwoKNNIAfter = comboTwoKNNI.fit_transform(X)
#print(comboTwoKNNIAfter.isnull().sum())
convertion = pd.DataFrame(comboTwoKNNIAfter)
#print(convertion.isnull().sum())
#print(convertion)

#2 Implement Max-Min normalization
comboTwoMM = MinMaxScaler()
comboTwoMMAfter = comboTwoMM.fit_transform(comboTwoKNNIAfter)


"""
hi = dataset.loc[:,'(Num) Column 6']
print(hi.min())
print(hi.max())
#print(type(comboTwoMMAfter))
newhi = pd.DataFrame(comboTwoMMAfter)
#print(newhi)
newhi2 = newhi.loc[:,5]
print(newhi2.min())
print(newhi2.max())


hi2 = dataset.loc[:,'(Num) Column 3']
print(hi2.max())
print(hi2.min())
xxx= newhi.loc[:,2]
print(xxx.max())
print(xxx.min())
"""

#3 Implement Isolation forest
comboTwoIF = IsolationForest(random_state=0)

#splilt the data into two sets

a = pd.DataFrame(comboTwoMMAfter)

#print(a)

#Set 1
set_One_Training = a.loc[0:749,:]
set_One_Testing = a.loc[750:1499,:]
test1 = comboTwoIF.fit(set_One_Training)
ans1 = test1.predict(set_One_Testing)
print(ans1)
d = 0
for x in ans1:
    if x==-1:
        d+=1
print(d)
#print(set_One_Training.shape)
#print(set_One_Testing.shape)

#Set 2
set_Two_Training = a.loc[750:1499,:]
set_Two_Testing = a.loc[0:749,:]
test2 = comboTwoIF.fit(set_Two_Training)
ans2 = test2.predict(set_Two_Testing)
print(ans2)
z = 0
for x in ans2:
    if x==-1:
        z+=1
print(z)

yuu = a.loc[288:360,:]
area = test2.predict(yuu)
print(area)
print(type(area))
hey= 0
for x in area:
    if x==-1:
        hey+=1
print(hey)


# Remove outliers for X
final_processed_X = a.drop(a.index[[288,292,294,296,299,303,309,311,316,320,321,326,327]],
                           axis=0)

print(final_processed_X.shape)


#Remove outliers for y
convertToDF = pd.DataFrame(y)
print(type(convertToDF))
final_processed_y = convertToDF.drop(convertToDF.index[[288,292,294,296,299,303,309,311,316,320,321,326,327]],
                                     axis=0)
print(final_processed_y.shape)


#Implement Decision tree
dtc = DecisionTreeClassifier(random_state=0)
dtc_Result = cross_val_score(dtc,final_processed_X,final_processed_y,cv=10,scoring='f1')
print(dtc_Result.mean())




#Implement random forest
rfc = RandomForestClassifier(random_state=0)
rfc_Result = cross_val_score(rfc,final_processed_X,final_processed_y.values.ravel(),cv=10,scoring='f1')
print(rfc_Result.mean())


#Implemt KNN
knnc = KNeighborsClassifier()
knnc_Result = cross_val_score(knnc,final_processed_X,final_processed_y.values.ravel(),cv=10,scoring='f1')
print(knnc_Result.mean())



#Implemt naive bayes
comNB = ComplementNB()
comNB_Result = cross_val_score(comNB,final_processed_X,final_processed_y.values.ravel(),cv=10,scoring='f1')
print(comNB_Result.mean())


#Calculate the average here
sumValue = dtc_Result.mean()+rfc_Result.mean()+knnc_Result.mean()+comNB_Result.mean()
averageValue = sumValue/4

print("The average of combination 2 is"+str(averageValue))

#1st attempt: 0.934  2nd attempt:0.935  3rd attempt:0.933  final average:0.934