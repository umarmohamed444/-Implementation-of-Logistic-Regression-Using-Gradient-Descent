# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:Umar Mohamed E 
RegisterNumber:212220040173
~~~
import numpy as np
def hypothesis(X,theta):
    z=np.dot(theta,X.T)
    s=1/(1+np.exp(-(z)))
    return s
def cost(X,y,theta):
    y1=hypothesis(X,theta)
    c=-(1/len(x))*np.sum(y*np.log(y1)+(1-y)*np.log(1-y1))
    return c
def gradient_descent(X,y,theta,alpha,iterations):
    m=len(X)
    J=[cost(X,y,theta)]
    for i in range(iterations):
        h=hypothesis(X,theta)
        for i in range(len(X.columns)):
            theta[i]=theta[i]-(alpha/m)*np.sum((h-y)-*X.iloc[:,i])
            J.append(cost(X,y,theta))
            return J ,theta
        def predict(X,y,theta,alpha, iterations):
Automatic O
50%
70%
J,th-gradient_descent(X,y,theta,alpha, iterations)
h-hypothesis (X,theta)
y pred=[]
for i in range(len(h));
if h[i]>-0.5:
else:
return J,y_pred
y_pred.append(1)
y_pred.append(0)
import pandas as pd
data-pd.read_excel("Placement_Data.xlsx")
data.head()
data1=data copy()
data1=datal.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1 ["gender"]=le.fit_transform(data1 ["gender"])
~~~
*/
```

## Output:
![logistic regression using gradient descent](sam.png)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

