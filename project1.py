from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('CausesOfDeath_France_2001-2008.csv')

df = df.drop(columns=['Flag and Footnotes'], errors='ignore')

df['Value'] = df['Value'].replace(':', np.nan)
df['Value'] = df['Value'].str.replace(' ', '', regex=True)
df = df.dropna(subset=['Value'])
df['Value'] = df['Value'].astype(float)


df_year = df.groupby('TIME')['Value'].sum().reset_index()


X = df_year[['TIME']]
y = df_year['Value']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


pred_2009 = model.predict([[2009]])
print("Predicted deaths for 2009:", pred_2009[0])


print("R2 Score:", r2_score(y_test, y_pred))

plt.scatter(X, y, label="Actual")
plt.plot(X, model.predict(X), label="Regression Line")
plt.xlabel('Year')
plt.ylabel('Total Deaths')
plt.title('Linear Regression: Year vs Total Deaths')
plt.legend()
plt.show()


print(df.dtypes)
print(df.head())

