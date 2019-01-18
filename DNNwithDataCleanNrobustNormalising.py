# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 12:03:09 2019

@author: balag
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import explained_variance_score, mean_absolute_error, median_absolute_error
from sklearn import preprocessing
import tensorflow as tf

data = pd.read_csv("d13.csv")
print(data.shape)
#data = pd.read_csv("d12_v4.csv")
data['d'] = pd.to_datetime(data['epoch'], unit='s')
data['day_time'] = pd.DatetimeIndex(data['d']).floor('1H')
data.drop(['hour','month','year','windspd','winddir','day','epoch'], axis=1, inplace = True)
data = data.groupby(['day_time']).mean()
data.index = data.index.floor('1D')
data = data.groupby(['day_time']).mean()

robust_scaler = preprocessing.RobustScaler()
np_scaled = robust_scaler.fit_transform(data)
print(np_scaled[0])
df_normalized_robust = pd.DataFrame(np_scaled, columns=['Latitude', 'Longitude', 'no', 'no2', 'nox', 'o3', 'temp', 'wind_x_dir',
       'wind_y_dir'], index = data.index)
df_normalized_robust.head(10)

def fillPrevData(df, feature, N):  
    rows = df.shape[0]
    nth_prior_measurements = [None]*N + [df[feature][i-N] for i in range(N, rows)]
    col_name = "{}_{}".format(feature, N)
    df[col_name] = nth_prior_measurements
#features = ['o3','no','no2','nox']
features = ['o3']
df_tmp = df_normalized_robust
for feature in features:  
    for N in range(1, 4):
        fillPrevData(df_tmp, feature, N)
print(df_tmp.head())

df_tmp = df_tmp.iloc[:,:].apply(pd.to_numeric, errors='coerce') 
df_tmp = df_tmp.dropna()  
df_tmp.describe().T 

df_tmp.info()
X = df_tmp.iloc[:,7:]
y = df_tmp['o3']
X.index = df_tmp.index.values.astype('datetime64[D]').astype(int)
y.index = df_tmp.index.values.astype('datetime64[D]').astype(int)
#y.head()

X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, random_state=23) 
X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=23)

X_train.shape, X_test.shape, X_val.shape  
print("Training instances   {}, Training features   {}".format(X_train.shape[0], X_train.shape[1]))  
print("Validation instances {}, Validation features {}".format(X_val.shape[0], X_val.shape[1]))  
print("Testing instances    {}, Testing features    {}".format(X_test.shape[0], X_test.shape[1])) 

feature_cols = [tf.feature_column.numeric_column(col) for col in X.columns] 
regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,  
                                      hidden_units=[50, 50],
                                      model_dir='tf_wx_model')

def wx_input_fn(X, y=None, num_epochs=None, shuffle=True, batch_size=400):  
    return tf.estimator.inputs.pandas_input_fn(x=X,
                                               y=y,
                                               num_epochs=num_epochs,
                                               shuffle=shuffle,
                                               batch_size=batch_size)

evaluations = []  
STEPS = 400  
for i in range(100):  
    regressor.train(input_fn=wx_input_fn(X_train, y=y_train), steps=STEPS)
    evaluations.append(regressor.evaluate(input_fn=wx_input_fn(X_val,
                                                               y_val,
                                                               num_epochs=1,
                                                               shuffle=False)))

plt.rcParams['figure.figsize'] = [14, 10]

loss_values = [ev['loss'] for ev in evaluations]  
training_steps = [ev['global_step'] for ev in evaluations]

plt.scatter(x=training_steps, y=loss_values)  
plt.xlabel('Training steps (Epochs = steps / 2)')  
plt.ylabel('Loss (SSE)')  
plt.show()  

pred = regressor.predict(input_fn=wx_input_fn(X_test,  
                                              num_epochs=1,
                                              shuffle=False))
predictions = np.array([p['predictions'][0] for p in pred])

print("The Explained Variance: %.2f" % explained_variance_score(  
                                            y_test, predictions))  
print("The Mean Absolute Error: %.2f degrees Celcius" % mean_absolute_error(  
                                            y_test, predictions))  
print("The Median Absolute Error: %.2f degrees Celcius" % median_absolute_error(  
                                            y_test, predictions))
print("Predicted Values:",predictions)