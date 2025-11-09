from tkinter import _test
from sklearn.preprocessing import LabelEncoder
from  sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df =  pd.read_csv('CausesOfDeath_France_2001-2008.csv')
df_label =df.copy()
le =LabelEncoder()
df_label['SEX'] = le.fit_transform(df_label['SEX'])
print(df_label)
df = df.drop(columns=['Flag and Footnotes'])
df['Value'] = df['Value'].replace(':', np.nan)
df['Value'] = df['Value'].str.replace(' ', '', regex=True)
df = df.dropna(subset=['Value'])
df['Value'] = df['Value'].astype(float)
print(df.isnull().sum())
X =df[['TIME']]
y =df['Value']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
pred_2009 = model.predict([[2009]])
print("Predicted deaths for 2009:", pred_2009[0])
print(y_pred)
print("R2 Score:", r2_score(y_test, y_pred))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Year')
plt.ylabel('Number of Deaths')
plt.title('Linear Regression: Year vs Deaths')
plt.legend()
plt.show()



print(df.dtypes)
print(df.head())


