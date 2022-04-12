#1.Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#2.Importing data

df=pd.read_csv(r'https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/MPG.csv')
df.head()

df.nunique()

df.columns

#3 Data Preprocessing

df.info()


df.describe()

df.corr()

#3.1 Missing Values

df = df.dropna()


df.info()

#4.DataVisualization

sns.pairplot(df,x_vars=['displacement', 'horsepower', 'weight','acceleration'])

#5.Defining target variables and features


y=df['mpg']

y.shape

X=df[['displacement', 'horsepower', 'weight','acceleration']]

X

#6.Scaling the data


from sklearn.preprocessing import StandardScaler


ss=StandardScaler()

X

X=ss.fit_transform(X)

X

X

pd.DataFrame(X).describe()

#7.Train test split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7,random_state=231)

X_train

X_train.shape,X_test.shape,y_train.shape,y_test.shape

#8.Model Selection(LinearRegression)

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

#9.Activating mode i.e fitting the model

lr.fit(X_train,y_train)

lr.intercept_


lr.coef_

#9.Predicting

y_pred=lr.predict(X_test)

y_pred

#8.Finding Accuracy

from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,r2_score

mean_absolute_error(y_test,y_pred)

mean_absolute_percentage_error(y_test,y_pred)

r2_score(y_test,y_pred)



#Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures

poly=PolynomialFeatures(degree=2)

X_train2=poly.fit_transform(X_train)

X_test2=poly.fit_transform(X_test)

lr.fit(X_train2,y_train)

lr.intercept_

lr.coef_

y_pred2=lr.predict(X_test2)

y_pred2


#Accuracy


from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,r2_score

mean_absolute_error(y_test,y_pred2)

mean_absolute_percentage_error(y_test,y_pred2)

r2_score(y_test,y_pred2)

