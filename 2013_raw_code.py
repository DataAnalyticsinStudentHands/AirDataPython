# -*- coding: utf-8 -*-
"""
Created on Tue May 22 14:55:45 2018

@author: Chinmay Dalvi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick_ohlc
from pandas import Series, DataFrame
from scipy import stats
import matplotlib as mpl
import seaborn as sns
%matplotlib qt

#Importing Data
filename = '2013_raw.csv'
ozone_data = pd.read_csv(filename, nrows=1000000, low_memory=False)

#Summary Statistics
ozone_data.describe()
ozone_data.info()

##Analysis Based on Time
#Unique Values in each column#Printing data
ozone_columns = ozone_data.columns.values.tolist()
for column_name in ozone_columns:
    print(ozone_data[column_name].unique())
    
#Unique Values in each column#Inputting unique values in DataFrame
unique_list = []
for column_name in ozone_columns:
    unique_list.append([ozone_data[column_name].unique()]) 
print(unique_list)
ozone_columns = pd.DataFrame(ozone_data.columns.values)
Unique_frame = pd.DataFrame(unique_list)
Unique_frame_2 = pd.concat([ozone_columns, Unique_frame], ignore_index = True, axis = 1)
Unique_frame_2.columns=['Column name', 'List of Unique values']
print(Unique_frame_2)

#Conversion of epoch Time to Actual Time
epoch_time = ozone_data['epoch']
Actual_DateTime = pd.to_datetime(epoch_time, unit='s')
Actual_DateTime = DataFrame(Actual_DateTime)
Actual_DateTime.rename(columns={'epoch':'actual'}, inplace =True)

# Histogram Generation
ozone_value = DataFrame(ozone_data, columns=['value'])
ozone_value.dropna(inplace=True)
y = np.arange(ozone_value.min(), ozone_value.max(), 100)
plt.hist(ozone_value['value'], bins=50, range=(0,1500), grid=True)
plt.figure(figsize=(10,8))
plt.grid(True)
plt.hist(ozone_value['value'], bins=1000, rwidth=0.8)

#Binning the data
Data_1 = ozone_value.unstack()
Bins = np.arange(Data_1.min(), Data_1.max(), 100)
categories = pd.cut(Data_1, Bins)
freq = pd.value_counts(categories, sort=False)
freq_data = pd.DataFrame(freq, columns = ['Frequency'])
freq_data.plot(kind = 'bar')
ozone_value['value'].min()
ozone_value['value'].max()
sns.distplot(ozone_value['value'],kde=False, bins =20)

# Missing data/null
N = ozone_data.isnull()
N.sum()
N_bool = N.replace([True, False],[1, 0])

# Input of Dates in data
ozone_data_date = pd.concat([Actual_DateTime, ozone_data], axis=1)
ozone_data_date['day'] = ozone_data_date['actual'].dt.day
ozone_data_date['weekday'] = ozone_data_date['actual'].dt.weekday
ozone_data_date['week'] = ozone_data_date['actual'].dt.week
ozone_data_date['weekday'].replace([1,2,3,4,5,6,0], ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], inplace=True)
ozone_data_date['month'] = ozone_data_date['actual'].dt.month
ozone_data_date['month_name'] = ozone_data_date['month']
ozone_data_date['month_name'].replace([1,2,3,4,5,6,7,8,9,10,11,12], ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], inplace=True)
ozone_data_date['year'] = ozone_data_date['actual'].dt.year
ozone_data_date['value_missing'] = N_bool['value']
ozone_data_date['flag_missing'] = N_bool['flag']

#attempt at functions
#def missing_analysis(timetype, plottype):
    #if timetype == 'month':
        #missing_group = ozone_data_date.groupby(['month', 'year'], as_index=False)
        #missing_data = missing_group.sum()
        #missing_data['Time_Period'] = missing_data['month'].astype(str) + [', '] + missing_data['year'].astype(str)
    #elif timetype == 'week':
        #missing_group = ozone_data_date.groupby(['week', 'year'], as_index=False)
        #missing_data = missing_group.sum()
        #missing_data['Time_Period'] = missing_data['week'].astype(str) + [', '] + missing_data['year'].astype(str)
    #elif timetype == 'day':
        #missing_group = ozone_data_date.groupby(['year', 'month', 'day'], as_index = False) 
        #missing_data = missing_group.sum()
        #missing_data['Time_Period'] = missing_data['year'].astype(str) + [', '] + missing_data['month'].astype(str) + [', '] + missing_data['day'].astype(str)
            ##Alternate method to be checked
            ##missing_data['Time_Period'] = pd.to_datetime(missing_data[['month','day','year']], format = '%m/%d/%Y')
            ##missing_data['Time_Period'] = missing_data['Time_Period'].dt.date
    #elif timetype == 'weekday':
        #missing_group = ozone_data_date.groupby(['weekday', 'year'], as_index=False)
        #missing_data = missing_group.sum()
        #missing_data['Time_Period'] = missing_data['weekday'].astype(str) + [', '] + missing_data['year'].astype(str)
    #elif timetype == 'year':
        #missing_group = ozone_data_date.groupby(['year'], as_index=False)
        #missing_data = missing_group.sum()
        #missing_data['Time_Period'] = missing_data['year']
    #else:
        #print('Invalid input') 
    #if plottype == 'line':
        #plt.plot(missing_data['Time_Period'], missing_data['value_missing'], missing_data['flag_missing'])
    #elif plottype == 'bar':
        #missing_data.plot(kind=plottype, x='Time_Period', y=['value_missing', 'flag_missing'])
    #else:
        #print('plot not supported yet')
    #return missing_data
       
## ##
#Plot of missing data over time period of Months
missing_group_month = ozone_data_date.groupby(['month','year'], as_index=False)
missing_month = missing_group_month.sum()
missing_month['day'] = 1 #dummy day

missing_month['date'] = pd.to_datetime(missing_month[['month', 'day','year']], format = '%m/%d/%Y')
missing_month['period'] = missing_month.date.dt.to_period('M') #Conversion to month, year format
missing_month.plot(x = 'period', y=['value_missing', 'flag_missing'], figsize=(12,6))
missing_month.plot(kind = 'bar', x = 'period', y=['value_missing', 'flag_missing'], figsize=(12,6))

#Codes attempted for generating plot
#plt.title('Histogram')
#plt.xlabel('Time Period')
#plt.ylabel('Missing Values')
#plt.grid(True)
#plt.xticks(missing_month.period)
#x = list(missing_month.period)
#y = list(missing_month.value_missing)
#plt.plot(x, y)
#missing_month.plot(kind = 'bar', y=['value_missing', 'flag_missing'], figsize=(12,6))
#missing_month.plot(x=['period'], y=['value_missing', 'flag_missing'], figsize=(12,6))
#plt.plot(missing_month['period'], missing_month['value_missing'])

#Plot of missing data over time period of Weeks
missing_group_week = ozone_data_date.groupby(['week','year'], as_index=False)
missing_week = missing_group_week.sum()
missing_month
missing_week.plot(kind = 'bar', x=['week', 'year'],y=['value_missing', 'flag_missing'], figsize=(12,6))
missing_week.plot(x=['week', 'year'],y=['value_missing', 'flag_missing'], figsize=(12,6))
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
# fig.subplots_asjust(botton=0.2)
ax1.plot(missing_week['week'], missing_week['flag_missing'], label='Flag missing')
ax1.plot(missing_week['week'], missing_week['value_missing'], label = 'Value missing')
ax1.legend(loc='best')
ax1.set_xlabel(r"Week")
#ax1.set_xticks(missing_week['weak'])
#ax1.set_xticklabels(missing_week['year'], minor = True)
ax1.grid(True)
ax2.xaxis.set_ticks_position("bottom")
ax2.xaxis.set_label_position("bottom")
ax2.spines["bottom"].set_position(("axes",-0.08))#shifts the 2nd axis down by a few points
ax2.set_xticks(missing_week['year'])
ax2.set_xlabel(r"Year")
plt.show()


#Plot of missing data over time period of Days; Method 1(does not create a tuple in the index)
missing_group_day = ozone_data_date.groupby(['month','day','year'], as_index =False)
missing_day = missing_group_day.sum()
missing_day['date'] = pd.to_datetime(missing_day[['month', 'day','year']], format = '%m/%d/%Y') ###
missing_day['date'] = missing_day['date'].dt.date
missing_day.plot(kind = 'bar', x = 'date', y=['value_missing', 'flag_missing'], figsize=(12,6))
missing_day.plot(x = 'date', y=['value_missing', 'flag_missing'], figsize=(12,6))

#Plot of missing data over time period of Days; Method 2, (resets the index)
missing_group_day_m2 = ozone_data_date.groupby(['month','day','year'])
missing_day_m2 = missing_group_day_m2.sum()
missing_day_m2.plot(kind ='bar', y=['value_missing', 'flag_missing'], figsize=(12,6))
missing_day_m2.reset_index(inplace=True)
missing_day_m2['date'] = pd.to_datetime(missing_day_m2[['month','day','year']], format = '%m/%d/%Y')
missing_day_m2['date'] = missing_day_m2['date'].dt.date
missing_day_m2.plot(x = 'date', y=['value_missing', 'flag_missing'], figsize=(12,6))

#analysis of missing values over regions
ozone_data_loc = ozone_data
ozone_data_loc['value_missing'] = N_bool['value']
ozone_data_loc['flag_missing'] = N_bool['flag']
ozone_data_loc.drop(['value','flag'], axis =1, inplace = True)

#construction of function for grouping on basis of required variable
def ozone_data_group(variable_type):
    ozone_data_group = ozone_data_loc.groupby([variable_type])
    ozone_data_statistic = ozone_data_group.sum()
    ozone_data_statistic.plot(kind='bar', y=['value_missing', 'flag_missing'])
    return ozone_data_statistic

#Assignment of statistics for each group 
ozone_data_statistic_site = ozone_data_group('site')
ozone_data_statistic_site = ozone_data_group('region')
ozone_data_statistic_site = ozone_data_group('cams')

#Grouping separately for region; #attempt 1    
#ozone_data_region_group = ozone_data_loc.groupby(['region'])
#ozone_data_region = ozone_data_region_group.sum()
#ozone_data_region.plot(kind='bar', y=['value_missing', 'flag_missing']) 

#grouping on basis of region
#ozone_data_region_group = ozone_data_loc.groupby(['region'])
#ozone_data_region = ozone_data_region_group.sum()
#ozone_data_region.plot(kind='bar', y=['value_missing', 'flag_missing']) 