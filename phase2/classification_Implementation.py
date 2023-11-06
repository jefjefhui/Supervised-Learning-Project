import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import f1_score,accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import GaussianNB
import csv


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
DTCResult = cross_val_score(DTC,final_processed_X,final_processed_y,cv=10,scoring='f1')
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


#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#The above codes are from combination 3
#I will implement the classification techniques below

#1 Decision tree classifer(After preprocessing)
dtcImplement = DecisionTreeClassifier(random_state=0)
#dtcImplementResult = cross_val_score(dtcImplement,final_processed_X,final_processed_y,scoring='f1', cv=10)
#print(dtcImplementResult.mean())



#2 Random Forest classifier (After preprocessing)
#rfcImplement =GridSearchCV(RandomForestClassifier(),{
#'n_estimators':[90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110],
#'max_features':[2,3,4,5,6,7,8,9,10,11]
#},cv=10,scoring='f1')

#rfcImplement.fit(final_processed_X,final_processed_y.values.ravel())
#rfcImplement.cv_results_
#print(pd.DataFrame(rfcImplement.cv_results_))
#print(rfcImplement.best_score_)
#print(rfcImplement.best_params_)




#3 KNN classifier (After preprocessing)
#knncImplement =GridSearchCV(KNeighborsClassifier(),{
 #   'n_neighbors':[3,5,7,9,11,13,15]
#},cv=5,scoring='f1')

#knncImplement.fit(final_processed_X,final_processed_y.values.ravel())
#print(pd.DataFrame(knncImplement.cv_results_))
#print(knncImplement.best_score_)
#print((knncImplement.best_params_))




#4 naive bayes classifier (After preprocessing)
comNBImplement = ComplementNB()
#comNBImplementOutcome = cross_val_score(comNBImplement,final_processed_X,final_processed_y.values.ravel(),scoring='f1',cv=10)
#print(comNBImplementOutcome.mean())


#gauNBImplement = GaussianNB()
#gauNBImplementOutcome = cross_val_score(gauNBImplement,final_processed_X,final_processed_y.values.ravel(),scoring='f1',cv=5)
#print(gauNBImplementOutcome.mean())

#catNBImplement = CategoricalNB()
#catNBImplementOutcome = cross_val_score(catNBImplement,final_processed_X,final_processed_y.values.ravel(),scoring='f1',cv=5)
#print(catNBImplementOutcome.mean())




#final selected KNN
finalKNNModel = KNeighborsClassifier(n_neighbors=5)


#the ensemble classifier (Voting)
votingImplement = VotingClassifier(estimators=[
    ('knn',finalKNNModel),('dt',dtcImplement),('cnb',comNBImplement)
])
votingImplementOutcome = cross_val_score(votingImplement,final_processed_X,final_processed_y.values.ravel(),scoring='f1',cv=10)
print("The number below is the final F1 score in the report:")
print(votingImplementOutcome.mean())


print("The number below is the final accuracy score in the report:")
theFinalAccuracy = cross_val_score(votingImplement,final_processed_X,final_processed_y.values.ravel(),cv=10)
print(theFinalAccuracy.mean())



#Ensemble classifier has the highest score among the 5 classsifers, which is the selected final model.


#Read the test data file
test_Data = pd.read_csv("Ecoli_test.csv")
#print(type(test_Data))
finalTestingData = test_Data.loc[:,:]
print(finalTestingData.shape)

#print(test_Data.isnull().sum())

MMSForTestData = MinMaxScaler()
MMSForTestDataAfter =MMSForTestData.fit_transform(finalTestingData)


#finalAnswer = votingImplement.predict(MMSForTestDataAfter)


#finalAccuracy = cross_val_score()






guessPrediction = VotingClassifier(estimators=[
    ('knn',finalKNNModel),('dt',dtcImplement),('cnb',comNBImplement)
])

abcde = guessPrediction.fit(final_processed_X,final_processed_y.values.ravel())

abcdeResult = abcde.predict(MMSForTestDataAfter)

print("The large array below is the prediction of Ecoli_test.csv data:")
print(abcdeResult)


#with open('s4658759.csv','w',newline='') as f:
 #   writer = csv.writer(f)
  #  for x in abcdeResult:
   #     writer.writerows(x)


checkfinal= 0
for x in abcdeResult:
    if x==1:
        checkfinal+=1
print("The number below shows how many 1s in the prediction of the Ecoli_test.csv:")
print(checkfinal)