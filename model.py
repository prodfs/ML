# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 13:37:07 2020

@author: hp
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection,neighbors,metrics,svm,preprocessing
import pickle

#step1:Import csv
dataset=pd.read_csv('D:\\ml\\New folder\\datasets\\Social_Network_Ads.csv')
dataset.head()

#step2:cleaning
dataset.isna().any() # no null value so no need to clean it
#drop unnecessary columns
dataset.drop('User ID',axis=1,inplace=True)

#Slicing
x=dataset.iloc[:,1:3]
#x1=dataset.drop('Purchased',axis=1)
y=dataset.iloc[:,3]

#train
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.2,random_state=0)

#feature selection to scale down values
scx=preprocessing.StandardScaler()
x_train=scx.fit_transform(x_train)
x_test=scx.transform(x_test)

#model creation
classifier=svm.SVC(kernel='linear',random_state=0)
classifier.fit(x_train,y_train)

#predict the test results
y_pre=classifier.predict(x_test)


#confusion matrix
cm=metrics.confusion_matrix(y_test,y_pre)
makeup=plt.subplot()
sns.heatmap(cm,annot=True,ax=makeup)
makeup.set_title('Confusion matrix')
makeup.set_xlabel('Predict')
makeup.set_ylabel('Actual')
plt.plot


#visualization the test result
from matplotlib.colors import ListedColormap
from matplotlib import style
x_set,y_set=x_test,y_test
#mashgrip,contourf
X1,X2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
                  np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
plt.figure(figsize=(16,9))
style.use('ggplot')
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75,cmap=ListedColormap(('r','g')))
plt.xlim(X1.min())
plt.ylim(X2.min())
#scatterplot
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(('b','k'))(i),label=j)
plt.title('Test result',fontsize='20')
plt.xlabel('Age',fontsize='20')
plt.ylabel('Estimated salary',fontsize='20')
plt.legend()
plt.show()

#saving model to disk
pickle.dump(classifier,open('model.pkl','wb'))

#loading model to compare the result
model=pickle.load(open('model.pkl','rb'))
model.predict([[0,5]])





