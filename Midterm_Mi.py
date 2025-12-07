import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

df=pd.read_csv('basic_edu.csv')
df = df[df["Code"]!='OWID_WRL']
encoder=LabelEncoder() 
df['Code']=encoder.fit_transform(df['Code']) + 1
df
x=df.drop(columns=['Entity', 'Share of population with at least some basic education'])
y=df['Share of population with at least some basic education']
x
y
X_train,X_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,random_state=42)
X_train.shape,y_train.shape
X_test.shape,y_test.shape
lr=LinearRegression()
lr
lr.fit(X_train,y_train)
results=lr.predict(X_test)
results
mse=mean_squared_error(y_test, results)
mse
mse=mean_squared_error(y_test, results)
mse
r2 = r2_score(y_test, results)
r2 

print(r2)
print("Saving model is done.")

