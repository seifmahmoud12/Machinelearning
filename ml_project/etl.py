import pymysql as pymysql
connection =pymysql.connect(host="127.0.0.1", port=3306, user="root", passwd="", database="BI_project",    autocommit=True)
cursor = connection.cursor()
import csv
from datetime import datetime
import pandas as pd
empdata = pd.read_csv('destree_predicted_values.csv', index_col=False, delimiter = ',')



for i, row in empdata.iterrows():


            query = """INSERT INTO destree(Actual,Predicted) VALUES (%s, %s)"""

            cursor.execute(query, tuple(row))
            print("Record inserted")





connection.close()