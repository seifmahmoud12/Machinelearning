import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score


df2=pd.read_csv("Temp_and_rain.csv")

print(df2.head())


d=df2.dropna()
X = df2.drop('Weather', axis=1)

y = df2['temP']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
l = LinearRegression()
l.fit(X_train, y_train)
l_pred = l.predict(X_test)
l_mse = mean_squared_error(y_test, l_pred)

print('Linear Regression: ', l_mse)
#-----------------------------------------#
r = RandomForestRegressor()
r.fit(X_train, y_train)
r_pred = r.predict(X_test)
r_mse = mean_squared_error(y_test, r_pred)
print('Random Forest ', r_mse)


