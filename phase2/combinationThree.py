import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
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


#Implementing combination 3(Isolation forest, Max-Min Normalization,Iterative imputer )
#1 Implement iterative imputer
II = IterativeImputer()
IIAfter = II.fit_transform(X)
#convert = pd.DataFrame(IIAfter)
#print(convert.isnull().sum())



#2Implement Max-min normalization
MMS = MinMaxScaler()
MMSAfter = MMS.fit_transform(IIAfter)

MMSAftertoDF = pd.DataFrame(MMSAfter)

#print(X)
#checking = X.loc[:,"(Num) Column 6"]
#print(checking.max())
#print(checking.min())
#print((MMSAftertoDF))
#checking2 = MMSAftertoDF.loc[:,5]
#print(checking2.max())
#print(checking2.min())


#checking3 = X.loc[:,"(Num) Column 3"]
#print(checking3.max())
#print(checking3.min())
#checking4 = MMSAftertoDF.loc[:,2]
#print(checking4.max())
#print(checking4.min())


#3 Implement isolation forest
IF = IsolationForest(random_state=0)

#Split the data to 2 sets
h = pd.DataFrame(MMSAfter)

#set 1
set1_Training = h.loc[0:749,:]
set1_Testing = h.loc[750:1499,:]
test1 = IF.fit(set1_Training)
test1_Result = test1.predict(set1_Testing)
print(test1_Result)
e= 0
for x in test1_Result:
    if x==-1:
        e+=1
print(e)

#Set 2
set2_Training = h.loc[750:1499.:]
set2_Testing = h.loc[0:749,:]
test2 =IF.fit(set2_Training)
test2_Result = test2.predict(set2_Testing)
print(test2_Result)
yy = 0
for x in test2_Result:
    if x==-1:
        yy+=1
print(yy)

#find the location of the outliers
#between 288-336
#296,298,322
#toDF = pd.DataFrame(test2_Result)
#actual_location = toDF.loc[288:336,:]
#print(actual_location)
#actual_location= test2_Result[288:336]
#print(actual_location)
print(type(test2_Result))
print(test2_Result[295])
print(test2_Result[297])
print(test2_Result[321])


#remove outliers
#Remove outlier for X
final_processed_X = h.drop(h.index[[295,297,321]],
                           axis=0)

print(final_processed_X.shape)


#Remove outlier for y
#print(type(y))
#print(y.shape)
convertytoDF = pd.DataFrame(y)
print(type(convertytoDF))
print(convertytoDF.shape)

final_processed_y = convertytoDF.drop(convertytoDF.index[[295,297,321]],
                                      axis=0)

print(final_processed_y.shape)

print('/////////////////////////////////////////////////////////////////////////////////////////////////////////')

#Implement Decision Tree
DTC = DecisionTreeClassifier(random_state=0)
DTCResult = cross_val_score(DTC,final_processed_X,final_processed_y,cv=10, scoring='f1')
print(DTCResult.mean())





#Implement Random forest
rfc = RandomForestClassifier(random_state=0)
rfcResult = cross_val_score(rfc,final_processed_X,final_processed_y.values.ravel(),cv=10,scoring='f1')
print(rfcResult.mean())




#Implement KNN
knnc = KNeighborsClassifier()
knncResult = cross_val_score(knnc,final_processed_X,final_processed_y.values.ravel(),cv=10,scoring='f1')
print(knncResult.mean())




#Implement Naive Bayes
comNB = ComplementNB()
comNBResult = cross_val_score(comNB,final_processed_X,final_processed_y.values.ravel(),cv=10,scoring='f1')
print(comNBResult.mean())



#calculate the average
sum = DTCResult.mean()+rfcResult.mean()+knncResult.mean()+comNBResult.mean()
average = sum/4

print('the average of combination 3 is'+str(average))


#1st attempt: 0.952  2nd attempt:0.952  3rd Attempt:0.951   final average:0.952