import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
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


#Implementation of combination 6(Isolation forest, Standardization,Iterative Imputer )
#1 Implement Iterative Imputer
ii = IterativeImputer()
iiAfter = ii.fit_transform(X)

vv = pd.DataFrame(iiAfter)

#print(vv.isnull().sum())
#print(X.isnull().sum())

#2 Implement Standardization
sad = StandardScaler()
sadAfter = sad.fit_transform(iiAfter)


#ppp = X.loc[:,'(Num) Column 6']
#print(ppp.max())
#print(ppp.min())

#kkk=pd.DataFrame(sadAfter)
#lll =kkk.loc[:,5]
#print(lll.max())
#print(lll.min())


#mmm = X.loc[:,'(Num) Column 3']
#print(mmm.max())
#print(mmm.min())

#nnn = kkk.loc[:,2]
#print(nnn.max())
#print(nnn.min())


#3 Implement isolation forest
IF = IsolationForest(random_state=0)


#convert to DF
cc = pd.DataFrame(sadAfter)

#Split the data into two sets

set1Training = cc.loc[0:749,:]
set1Testing =cc.loc[750:1499,:]
test1 = IF.fit(set1Training)
test1Result = test1.predict(set1Testing)
print(test1Result)
ty =0
for x in test1Result:
    if x==-1:
        ty+=1
print(ty)

set2Training = cc.loc[750:1499,:]
set2Testing = cc.loc[0:749,:]
test2 = IF.fit(set2Training)
test2Result = test2.predict(set2Testing)
print(test2Result)
hi=0
for x in test2Result:
    if x==-1:
        hi+=1
print(hi)


#looking for the actual position of the outlier
#296,298,322
print(test2Result[295])
print(test2Result[297])
print(test2Result[321])



#Remove outliers for X
final_processedX = cc.drop(cc.index[[295,297,321]],
                           axis=0)
print(final_processedX.shape)


#Remove outliers for y
#print(type(y))
convertytodf = pd.DataFrame(y)
#print(type(convertytodf))

#print(convertytodf)

final_processedy = convertytodf.drop(convertytodf.index[[295,297,321]],
                                     axis=0)

print(final_processedy.shape)


#Implement Decision Tree
dtc = DecisionTreeClassifier(random_state=0)
dtcResult = cross_val_score(dtc,final_processedX,final_processedy,cv=10,scoring='f1')
print(dtcResult.mean())




#Implement Random Forest
rfc = RandomForestClassifier(random_state=0)
rfcResult = cross_val_score(rfc,final_processedX,final_processedy.values.ravel(),cv=10,scoring='f1')
print(rfcResult.mean())


#Implement KNN
knnc = KNeighborsClassifier()
knncResult = cross_val_score(knnc,final_processedX,final_processedy.values.ravel(),cv=10,scoring='f1')
print(knncResult.mean())



#Implement Naive Bayes
gauNB = GaussianNB()
gauNBResult = cross_val_score(gauNB,final_processedX,final_processedy.values.ravel(),cv=10,scoring='f1')
print(gauNBResult.mean())



#Calculate the average of combination 6
sum = dtcResult.mean()+rfcResult.mean()+knncResult.mean()+gauNBResult.mean()
average = sum/4

print("The average score for combination 6 is "+ str(average))



# 1st attempt: 0.749  2nd attempt:0.749   3rd attempt: 0.750   final average: 0.749