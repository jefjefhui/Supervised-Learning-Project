INFS7203 Data Mining Project Phase 2 instructions

Name:Chi Heng Jeffrey Hui
Student number: 46587590

Operation system: Windows 10 Home
Programming language: Python 3.9.7
Installed packages: Scikit-learn, pandas, numpy
IDE: PyCham 


Reminder before reproduce the codes
-Install all the required packages before running the codes 
-Make sure all the python files, Ecoli.csv and Ecoli_test.csv are in the same file. It is important to place all of them in same file, so that python can read the csv and work on the models.
-Make sure you don't edit the csv file names


Explanation of each file:
1. main.py: main.py implements the first combination of of preprocessing techniques, which is isolation forest, Max-min normiazlaition, and SimpleImputer. 
2. combinationTwo.py: combinationTwo.py implements the second combination of preprocessing techniques, which is isolation forest, Max-min normalization and KNNImputer. 
3. combinationThree.py: combinationThree.py implements the third combination of preprocessing techniques, which is isolation forest, max-min normalization, and iterative imputer.
4.combinationFour.py: combinationFour.py implements the forth combination of preprocessing techniques, which is islation forest, standardization, and simple imputer. 
5. combinationFive.py: combinationFive.py implments the fifth combination of preprocessing techniques, which is isolation forest, standardization and KNN imputer. 
6. combinationSix.py: combinationSix.py implements the sixth conbinations of preprocessing techniques, which is isolation forest, standardization, and iterative imputer. 
7. classification_Implementation.py: classification_Implementation.py uses the selected preprocessing techniques, and work on decision tree, random forest, KNN,naive bayes and the ensemble classifier. After picking the ensemble classifier as the final model, I also work on the F1 and accuracy socre in this python file. This file also use the model to predict the outcomes of the Ecoli_test.csv file.


Flow of my work:
Firsly, I work on main.py, conbinationTwo.py, combinationThree.py, combinationFour.py, combinationFive.py, and combinationSix.py. At the bottom of each of these python files, it shows the score for each combination. As combination 3 has the highest score, it is selected as the preprocessing techniques for this assignment.
Afterwards, I created classification_Implementation.py. From line 19 to 184 in classification_Implementation.py, these are the codes from combinationThree.py. From 189 to 192 in classification_Implementation.py, it is the work for decision tree. From line 196 to 206 in classification_Implementation.py, these are the codes for random forest classifier. From 211 to 219 in classification_Implementation.py, these are the codes for KNN classifier. From line 224 to 236 in classification_Implementation.py, these are the codes for naive bayes classifier. From line 245 to 260 in classfication_Implementation.py, these are the codes for ensemble classifier. From line 291 to 294 in classification_Implementation.py, these are the codes to predict the provided test data, and it also print out the result. From line 303 to 308 in classification_Implementation.py, these codes is to count how many 1 in the predicted outcome. 


Extra information: 
To help the grader see the results, I add serval texts. In the output window, you should see "The number below is the final F1 score in the report:", "The number below is the final accuracy score in the report:", and "The large array below is the prediction of Ecoli_test.csv data:". Under these texts, you should see the corresponding infomration. In addition, these are the information I put on the result report. 


Reproduce the code: 
When you unzip the compressed file, you will see this readme and a file folder called phase 2, this folder contains all the files I mentioned above. 
To reproduce my work, just follow the information above and press the run button in your IDE, it should work.     