import pandas as pd
from sklearn import model_selection,gaussian_process,multiclass,metrics
import numpy as np
import matplotlib.pyplot as plt

def classifier(x):
    dic= {2:0,3:0,4:0,5:1,6:1,7:2,8:2}
    return dic[x]

df = pd.read_csv("WineQT.csv")
for i in range(0,len(df)):
    df.loc[i,"Target"] = classifier(df.loc[i,'quality'])
targets = dict(df['Target'].value_counts())


tar0=df.loc[df['Target'] == 0]
tar1=df.loc[df['Target'] == 1]
tar2=df.loc[df['Target'] == 2]

chosen=[list(df)[0:5],list(df)[5:10]]
fig, ax = plt.subplots(2,6)
for a in range(2):
    for b in range(5):
        bins = np.linspace(min(df[chosen[a][b]]), max(df[chosen[a][b]]), 10)
        ax[a,b].hist(list(df[chosen[a][b]]), bins, histtype='step', stacked=True, color="blue" )
        ax[a,b].set_title(chosen[a][b])
plt.show()

chosen=list(df)[0:5]
chosen2=list(df)[5:11]
for i in range(len(chosen)):
    bins = np.linspace(min(df[chosen[i]]),max(df[chosen[i]]),10)
    ax[0,i].hist(list(tar0[chosen[i]]),bins, histtype='step', stacked=True,color= "blue",density=True,label='0')
    ax[0,i].hist(list(tar1[chosen[i]]),bins, histtype='step', stacked=True,color= "red",density=True,label='1')
    ax[0,i].hist(list(tar2[chosen[i]]),bins, histtype='step', stacked=True,color= "green",density=True,label='2')
    ax[0,i].set_title(chosen[i])
    ax[0,i].legend()


ax[0,5].bar([ str(i) for i in targets.keys()], targets.values(), color='g')
ax[0,5].set_title("Histogram of Y")

for i in range(len(chosen2)):
    bins = np.linspace(min(df[chosen2[i]]),max(df[chosen2[i]]),10)
    ax[1,i].hist(list(tar0[chosen2[i]]),bins, histtype='step', stacked=True,color= "blue",density=True,label='0')
    ax[1,i].hist(list(tar1[chosen2[i]]),bins, histtype='step', stacked=True,color= "red",density=True,label='1')
    ax[1,i].hist(list(tar2[chosen2[i]]),bins, histtype='step', stacked=True,color= "green",density=True,label='2')
    ax[1,i].set_title(chosen2[i])
    ax[1,i].legend()


X=df.drop(['Target','Id','quality'],axis=1)
y=df['Target']

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, train_size=0.8)

X_test=df.drop(['Target','Id','quality'],axis=1)   #Removable
y_test=df['Target']      #Removable
estimator = gaussian_process.GaussianProcessClassifier()
ovr = multiclass.OneVsRestClassifier(estimator)
ovr.fit(X_train,y_train)
print('ovr:\n'+ metrics.classification_report(ovr.predict(X_test), y_test))
ovr_cm = metrics.confusion_matrix(y_test, ovr.predict(X_test), labels=ovr.classes_)
ovr_matrix= metrics.ConfusionMatrixDisplay(confusion_matrix=ovr_cm,display_labels=ovr.classes_)
ovr_matrix.plot()

#ovr_matrix = metrics.plot_confusion_matrix(ovr, X_test, y_test,cmap=plt.cm.Blues,normalize='true')
#plt.show()
'''
metrics.RocCurveDisplay.from_estimator(clf, X_test, y_test)
plt.show()
'''
ovo = multiclass.OneVsOneClassifier(estimator)
ovo.fit(X_train,y_train)
print('ovo:\n'+ metrics.classification_report(ovo.predict(X_test), y_test))
ovo_cm = metrics.confusion_matrix(y_test, ovo.predict(X_test), labels=ovo.classes_)
ovo_matrix= metrics.ConfusionMatrixDisplay(confusion_matrix=ovo_cm, display_labels=ovo.classes_)
ovo_matrix.plot()
plt.show()
#ovo_matrix = metrics.plot_confusion_matrix(ovo, X_test, y_test,cmap=plt.cm.Blues,normalize='true')