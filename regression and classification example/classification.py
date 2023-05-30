import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.svm import SVC

df2=pd.read_csv("Temp_and_rain.csv")

print(df2.head())


d=df2.dropna()
X = df2.drop('Weather', axis=1)
y = df2['Weather']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
r = RandomForestClassifier()
r.fit(X_train, y_train)
r_pred = r.predict(X_test)
r_acc = accuracy_score(y_test, r_pred)

print('Random Forest: ', r_acc)
#--------------------------------------------------#
svm = SVC()
svm.fit(X_train, y_train)
svc_pred = svm.predict(X_test)
svc_acc = accuracy_score(y_test, svc_pred)
print("SVMAccuracy:", svc_acc)