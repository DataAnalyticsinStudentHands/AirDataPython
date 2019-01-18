# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 11:46:30 2019

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

# Call describe on df and transpose it due to the large number of columns
spread = df_tmp.describe().T

# precalculate interquartile range for ease of use in next calculation
IQR = spread['75%'] - spread['25%']

# create an outliers column which is either 3 IQRs below the first quartile or
# 3 IQRs above the third quartile
spread['outliers'] = (spread['min']<(spread['25%']-(3*IQR)))|(spread['max'] > (spread['75%']+3*IQR))

# just display the features containing extreme outliers
spread.iloc[:,:]

predictors = ['wind_x_dir','wind_y_dir','Longitude','Latitude','temp','o3_3','o3_2','o3_1']
df2 = df_tmp[['o3'] + predictors] 


# manually set the parameters of the figure to and appropriate size
plt.rcParams['figure.figsize'] = [16, 22]

# call subplots specifying the grid structure we desire and that 
# the y axes should be shared
fig, axes = plt.subplots(nrows=4, ncols=2, sharey=True)

# Since it would be nice to loop through the features in to build this plot
# let us rearrange our data into a 2D array of 6 rows and 3 columns
arr = np.array(predictors).reshape(4, 2)

# use enumerate to loop over the arr 2D array of rows and columns
# and create scatter plots of each meantempm vs each feature
for row, col_arr in enumerate(arr):  
    for col, feature in enumerate(col_arr):
        axes[row, col].scatter(df2[feature], df2['o3'])
        if col == 0:
            axes[row, col].set(xlabel=feature, ylabel='o3')
        else:
            axes[row, col].set(xlabel=feature)
plt.show()  

X = df2[predictors]  
y = df2['o3']

# Add a constant to the predictor variable set to represent the Bo intercept
X = sm.add_constant(X)  
X.iloc[:10, :10]  

# (1) select a significance value
alpha = 0.05

# (2) Fit the model
model = sm.OLS(y, X).fit()

# (3) evaluate the coefficients' p-values
model.summary()  

# first remove the const column because unlike statsmodels, SciKit-Learn will add that in for us


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12) 

regressor = LinearRegression()

# fit the build the model by fitting the regressor to the training data
regressor.fit(X_train, y_train)

# make a prediction set using the test set
prediction = regressor.predict(X_test)

# Evaluate the prediction accuracy of the model

print("The Explained Variance: %.2f" % regressor.score(X_test, y_test))  
print("The Mean o3 Error: %.2f" % mean_absolute_error(y_test, prediction))  
print("The Median o3 Error: %.2f" % median_absolute_error(y_test, prediction))  
print("Predicted Values:",prediction)