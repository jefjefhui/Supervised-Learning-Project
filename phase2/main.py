import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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


#Implement the first preprocessing combo: Isolation Forest,Max-Min normalization,and SimpleImputer
#1. Implement the SimpleImputer
comboOneSI = SimpleImputer(missing_values=np.nan,strategy='mean')
comboOneSIAfter = comboOneSI.fit_transform(X)

#Transform the numpy array to pandas dataframe and check if there are any nan values
#a = pd.DataFrame(comboOneSIAfter)
#print(a.isnull().sum())

#2 Implement the Max-Min normalization
#Check the previous values
#column = X["(Num) Column 3"]
#print(column.min())
comboOneMM = MinMaxScaler()
comboOneMMAfter =comboOneMM.fit_transform(comboOneSIAfter)

#Convert the numpy array to pandas dataframe
#b = pd.DataFrame(comboOneMMAfter)
#column2 = b[2]
#print(column2.max())
#print(column2.min())


#3 Implement Isolation Forest
#print(b.shape)
comboOneIF = IsolationForest(random_state=0)
#X_train, X_test = train_test_split(comboOneMMAfter)
#print(X_train.shape)
#print(X_test.shape)
#test1 = comboOneIF.fit(X_train)
#ans1 = test1.predict(X_test)
#print(ans1)

#Do 2 rounds of Isolation Forest

#Convert to pandas dataframe
c = pd.DataFrame(comboOneMMAfter)

#First test
first_train_set = c.loc[0:749, :]
first_test_set = c.loc[750:1500, :]
#print(first_test_set.shape)
test1 = comboOneIF.fit(first_train_set)
ans1 = test1.predict(first_test_set)
print(ans1)
t =0
for x in ans1:
   if x==-1:
        t+=1
print(t)

#Second test
second_train_set = c.loc[750:1500, :]
second_test_set = c.loc[0:749, :]
#print(second_train_set.shape)
#print(second_test_set.shape)
test2 = comboOneIF.fit(second_train_set)
ans2 = test2.predict(second_test_set)
print(ans2)
r = 0
for x in ans2:
    if x==-1:
        r+=1
print(r)

#Find out the location of the outliers
findOut = c.loc[288:336,:]
ans3 = test2.predict(findOut)
print(ans3)

#print(comboOneMMAfter.shape)

#rr = c.loc[317:328,:]
#ans4 = test2.predict(rr)
#print(ans4)
#print('//////////////////////////////////////')
#print(c)
#print(type(findOut))
#print(findOut)


#Remove Outliers for X
final_processedX = c.drop(c.index[[294,296,299,303,309,316,320,321,326,327]],
                          axis=0)

print(final_processedX.shape)


#Convert to dataframe and remove outliers of y
toDF = pd.DataFrame(y)

final_processedy = toDF.drop(toDF.index[[294,296,299,303,309,316,320,321,326,327]],
                          axis=0)
print(final_processedy.shape)



#Implementing the decision tree classifer
dtc = DecisionTreeClassifier(random_state=0)
dtc_Result = cross_val_score(dtc,final_processedX,final_processedy,cv=10,scoring='f1')
print(dtc_Result.mean())



#Implementing the random forest classifier
rfc = RandomForestClassifier(random_state=0)
rfc_Result = cross_val_score(rfc,final_processedX,final_processedy.values.ravel(),cv=10,scoring='f1')
print(rfc_Result.mean())

#print(final_processedy.shape)
#print(type(y))
#hihi = final_processedy.loc[:,:]
#print(type(hihi))
#print(hihi.shape)
#finish = final_processedy.values.ravel()
#print(type(finish))
#print(finish.shape)

#print('////////////////////////////////')
#print(final_processedX.shape)
#print(final_processedy.shape)
#print(finish.shape)



#Implementing the KNN classifier
knncl = KNeighborsClassifier()
knncl_Result = cross_val_score(knncl,final_processedX,final_processedy.values.ravel(),cv=10,scoring='f1')
print(knncl_Result.mean())




#Implementing the naive bayes
comNB = ComplementNB()
comNB_Result = cross_val_score(comNB,final_processedX,final_processedy.values.ravel(),cv=10,scoring='f1')
print(comNB_Result.mean())


#The average score of combination One (Isolation forest, Max-Min normalization and SimpleImputer )
sumValue = dtc_Result.mean()+rfc_Result.mean()+knncl_Result.mean()+comNB_Result.mean()
averageValue = sumValue/4
print(averageValue)

print("The average value of combination one is "+ str(averageValue))

#1st attempt: 0.933 2nd attempt:0.934  3rd attempt:0.935    final average: 0.934