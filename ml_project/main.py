
import pandas as pd
import sns as sns
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    mean_squared_error, r2_score
#df=pd.read_csv("health.csv")

# df[['LDL', 'HDL']] = df['Cholesterol Level'].str.split(',', expand=True)
# df["LDL"] = pd.to_numeric(df["LDL"])
# df["HDL"] = pd.to_numeric(df["HDL"])
# df['total Cholesterol'] = df['LDL'] + df['HDL']
# df.drop('Cholesterol Level',axis=1,inplace=True)
# df.to_csv("transHealth.csv",index=False)
data = pd.read_csv("transHealth.csv")
# import matplotlib.pyplot as plt
import seaborn as sns
#
# sns.scatterplot(x='BMI', y='total Cholesterol', hue='Gender', data=data)
# plt.title('BMI vs. Cholesterol by Gender')
# plt.show()
# plt.scatter(data['Age'], data['total Cholesterol'], c=data['Gender'].map({'Male': 'blue', 'Female': 'red'}))
# plt.title('Age vs. Cholesterol by Gender')
# plt.xlabel('Age')
# plt.ylabel('total Cholesterol')
# plt.show()
# #-----------------------------------#
# sns.countplot(x='Age', data=data)
# plt.title('Number of Patients by Age Group')
# plt.xlabel('Age Group')
# plt.ylabel('Number of Patients')
# plt.show()
#--------------------------------------------------------------#
#
#
#

# X = data[[ 'Age', 'total Cholesterol', 'Blood Pressure','BMI','Gender']]
# X_encoded_svc = pd.get_dummies(X)
# y = data['Diagnosis']
# X_train, X_test, y_train, y_test = train_test_split(X_encoded_svc, y, test_size=0.2)
#
#
# clf = SVC()
# clf.fit(X_train, y_train)
#
#
# y_pred1 = clf.predict(X_test)
# accuracy1 = accuracy_score(y_test, y_pred1)
# print('SVM Accuracy:',accuracy1)
# svm_predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred1})
#
#
# svm_predictions.to_csv('svm_diag_predicted_values.csv', index=False)


# #----------------------------------------------------------------------------------#####
#
#
# X = data[['Age', 'total Cholesterol', 'Blood Pressure','BMI','Gender']]
# X_encoded_ran = pd.get_dummies(X)
# y = data['Diagnosis']
# X_train1, X_test, y_train, y_test = train_test_split(X_encoded_ran, y, test_size=0.2)
# #
# #
# clf = RandomForestClassifier()
# clf.fit(X_train1, y_train)
#
#
# y_pred2 = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred2)
#
# # Print the accuracy score
# print('Random Forest Accuracy:', accuracy)
#
#
# random_predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred2})
#
# #Save the predicted values to a CSV file
# random_predictions.to_csv('random_class_predicted_values.csv', index=False)
# #----------------------------------------------------------------#
#
# cm1 = confusion_matrix(y_test, y_pred1)
# cm2 = confusion_matrix(y_test, y_pred2)
#
# # Calculate accuracy, precision, recall, and F1-score
# accuracy1 = accuracy_score(y_test, y_pred1)
#
#
# accuracy2 = accuracy_score(y_test, y_pred2)
#
#
# print("Confusion Matrix - Model 1:")
# print(cm1)
#
# print("\nConfusion Matrix - Model 2:")
# print(cm2)
#
# #---------------------------------------------------------------#
#
X = data[[ 'Age', 'Blood Pressure','BMI','Gender','Diagnosis']]
X_encoded_li = pd.get_dummies(X)
y = data['total Cholesterol']
X_train0, X_test0, y_train0, y_test0 = train_test_split(X_encoded_li, y, test_size=0.2, random_state=40)
l = LinearRegression()
l.fit(X_train0, y_train0)
l_pred = l.predict(X_test0)
l_mse = mean_squared_error(y_test0, l_pred)

print('Linear Regression: ', l_mse)

lin_predictions = pd.DataFrame({'Actual': y_test0, 'Predicted': l_pred})


lin_predictions.to_csv('linear_predicted_values.csv', index=False)
#
#
#
# dt_model = DecisionTreeRegressor()
# dt_model.fit(X_train0, y_train0)
# dt_pred = dt_model.predict(X_test0)
# print('Decision Tree Regression MSE:', mean_squared_error(y_test0, dt_pred))
# Des_predictions = pd.DataFrame({'Actual': y_test0, 'Predicted': dt_pred})
#
#
# Des_predictions.to_csv('destree_predicted_values.csv', index=False)
#
#
# #------------------------------------------------#
# X = data[['Age', 'Gender', 'Blood Pressure', 'total Cholesterol']]
# X_encoded_ = pd.get_dummies(X)
# # Scale the feature set
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_encoded_)
#
#
# dbscan = DBSCAN(eps=0.5, min_samples=5)
# dbscan.fit(X_scaled)
#
# # Add the cluster labels to the original dataset
# data['Cluster'] = dbscan.labels_
# #
# #
# plt.scatter(data['Age'], data['total Cholesterol'], c=data['Cluster'])
# plt.xlabel('Age')
# plt.ylabel('total Cholesterol')
# plt.show()
# #----------------------------------------------------#
#
#
# x = data[['Age', 'BMI', 'Blood Pressure', 'total Cholesterol']]
# X=pd.get_dummies(x)
#
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # Fit and predict clusters
# kmeans = KMeans(n_clusters=3, random_state=0)
# clusters = kmeans.fit_predict(X_scaled)
#
# # Add cluster labels to dataframe
# data['cluster'] = clusters
#
# plt.scatter(data['Age'], data['BMI'], c=data['cluster'])
# plt.title('Healthcare Dataset Clustering')
# plt.xlabel('Age')
# plt.ylabel('BMI')
# plt.show()

