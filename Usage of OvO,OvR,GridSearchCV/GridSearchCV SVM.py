import pandas as pd
import numpy as np
import sklearn
from sklearn import gaussian_process,multiclass,metrics
import statistics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import KFold, cross_val_score,RandomizedSearchCV, StratifiedKFold,train_test_split,GridSearchCV,cross_val_predict
from sklearn.svm import SVC
import itertools as it
import random

df = pd.read_csv("WaterQT.csv")
fig, ax = plt.subplots(2,5)

chosen=[list(df)[0:5],list(df)[5:10]]
tar0=df.loc[df['Potability'] == 0]
tar1=df.loc[df['Potability'] == 1]


#THIS IS FOR QUESTION A AND B
'''for a in range(2):
    for b in range(5):
        bins = np.linspace(min(df[chosen[a][b]]), max(df[chosen[a][b]]), 10)
        ax[a,b].hist(list(df[chosen[a][b]]), bins, histtype='step', stacked=True, color="blue" )
        ax[a,b].set_title(chosen[a][b])
plt.show()'''


for a in range(2):
    for b in range(5):
        bins = np.linspace(min(df[chosen[a][b]]), max(df[chosen[a][b]]), 10)
        ax[a,b].hist(list(tar0[chosen[a][b]]), bins, histtype='step', stacked=True, color="blue", density=True, label='0')
        ax[a,b].hist(list(tar1[chosen[a][b]]), bins, histtype='step', stacked=True, color="red", density=True, label='1')
        ax[a,b].set_title(chosen[a][b])
        ax[a,b].legend()

plt.show()


'''
Perform serval runs employing the SVM with different C and gamma parameter using cross 
validation. (no shuffle). Vary n_splits parameter between 10 and 50.
Display the confusion matrix of the test data for each run.
Calculate the obtained average Accuracy of the test data after 5 runs.
'''

X=df.drop(['Potability'],axis=1)
X= sklearn.preprocessing.StandardScaler().fit_transform(X)
y=df['Potability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 45)


#question c/d ( pre process)
clf = SVC()
param_grid =  {'C': [1, 100,10], 'gamma': [0.001, 0.01,0.1]}
#param_grid =  {'C': [1, 100,10], 'kernel':['linear']}
keys, values = zip(*param_grid.items())
permutations_dicts = [dict(zip(keys, v)) for v in it.product(*values)]
print(permutations_dicts)
random.shuffle(permutations_dicts)
accuracy_c,accuracy_d=[],[]

#This is for question c

for i in range(5):
    clf = SVC(**permutations_dicts[i])
    y_pred = cross_val_predict(clf, X, y, cv=10)
    conf_mat = confusion_matrix(y, y_pred)
    accuracy_c.append(accuracy_score(y,y_pred))
    print(conf_mat)
print("average accuracy: ",statistics.mean(accuracy_c))


#This is for question d
param_grid = [
    {'C': [1, 10, 100], 'kernel': ['linear']},
    {'C': [1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
 ]
for i in range(5):
    grid = GridSearchCV(SVC(), param_grid,refit = True).fit(X_train, y_train)
    print(grid.best_params_)
    predic = grid.predict(X_test)
    print(classification_report(y_test, predic))
    #print(predic,np.array(y_test))
    accuracy_d.append(accuracy_score(np.array(y_test), predic))
print("average accuracy: ",statistics.mean(accuracy_d))