# -*- coding: utf-8 -*-
"""
Created on Sun May 19 11:22:43 2019

@author: DELL
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('E:/Manika du/Kaggle/creditcard.csv')

target = df['Class']

features = df.iloc[:,0:30]

x = features
y=target

from sklearn.model_selection import train_test_split
#train_set, test_set = train_test_split(features, test_size=0.2, random_state=42) 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


from sklearn.preprocessing import StandardScaler

scaler_x = StandardScaler().fit(x_train)
rescaled_x = scaler_x.transform(x_train)



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

algorithm = LogisticRegression() 
algorithm.fit(x_train, y_train)

y_predictions = algorithm.predict(x_test)
algo_mse = mean_squared_error(y_test, y_predictions)
algo_rmse = np.sqrt(algo_mse)
print(" RMSE: ",algo_rmse)
score = algorithm.score(x_test,y_test)
print("Score: ",score)