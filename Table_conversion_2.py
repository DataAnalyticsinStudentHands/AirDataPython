# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 11:55:17 2018

@author: Chinmay Dalvi
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 10:19:14 2018

@author: Chinmay Dalvi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import matplotlib as mpl
import pickle
# import seaborn as sns
# %matplotlib qt

#Importing Data
filename = '2013_raw.csv'
original_data = pd.read_csv(filename, nrows=100000, low_memory=False) #creates a sample of the initial 100,000 rows of csv file, for loading entire csv file the nrows should be erased from the code

#Creation of location column consisting of region and sites
original_data['Location'] = original_data['region'].map(str) + [', '] + original_data['site'].map(str)

#Unique Values in each column#Inputting unique values in DataFrame
original_columns = original_data.columns.values.tolist() #creation of list of columns inside data
unique_list = []
for column_name in original_columns:   #creation of list of unique values in each column
    unique_list.append([original_data[column_name].unique()]) 
#print(unique_list)
original_columns = pd.DataFrame(original_data.columns.values) #making a dataframe consisting of each list
Unique_frame = pd.DataFrame(unique_list)
Unique_frame_Data = pd.concat([original_columns, Unique_frame], ignore_index = True, axis = 1) 
Unique_frame_Data.columns=['Column_name', 'List_of_Unique_values']
#for x in Unique_frame_Data.index: #if printing data is required
    #print(Unique_frame_Data.Column_name[x], Unique_frame_Data.List_of_Unique_values[x])

#Creation of Location dataframe showing list of sites belonging to which region
Location_df = pd.DataFrame(Unique_frame_Data.List_of_Unique_values[Unique_frame_Data['Column_name'] == 'Location'].iloc[0])
Location_df.rename(columns={0:'location'}, inplace =True)
Location_df = pd.DataFrame(Location_df.location.str.split(', ').tolist(), columns=['region','site'])
Location_df = Location_df.convert_objects(convert_numeric=True)

#Creation of list of epoch times
# Epoch_list = list(Unique_frame_Data.List_of_Unique_values[Unique_frame_Data['Column_name'] == 'epoch'].iloc[0])
# Start_time = min(Epoch_list)
Start_time = 1356998400 #start time for year 2013 i.e. 1st Jan 2013, 00:00 Hours
# Stop_time = max(Epoch_list)
Stop_time = 1388534100 #stop time for year 2013 31st Dec 2013, 23:55 Hours
Epoch = list(np.arange(Start_time, Stop_time+300, 300)) #Creation of list of epoch times at 5 min intervals

#Creation of list of parameters in original dataframe (which is to be inputted as new columns in new dataframe)
Parameters = list(Unique_frame_Data.List_of_Unique_values[Unique_frame_Data['Column_name'] == 'param_name'].iloc[0])
Para_Flags = Parameters
string = '_flags'
Para_Flags = [s + string for s in Para_Flags] # Creation of list of flags column for each parameter

#Creation of list of sites in entire dataframe
I = Unique_frame_Data.List_of_Unique_values[Unique_frame_Data['Column_name'] == 'site']
Site_List = sorted(list(I.iloc[0]))

pd.options.mode.chained_assignment = None # Ignoring error message regarding chained assignment

#loads the saved checkpoints
try: 
    data_new = pickle.load(open("data_converted.p", "rb"))
except:
    data_new = pd.DataFrame() # Creation of empty dataframe

try:
    Para_comp_check = pickle.load(open("Parameters_converted.p", "rb"))
    i = pickle.load(open("iteration_no.p", "rb"))
except:
    Para_comp_check = pd.DataFrame(Parameters, columns=['Parameter'])
    Para_comp_check['check']='Not_done'

try:
    i = pickle.load(open("iteration_no.p", "rb"))
except:
    i = 0 # considered as iteration no

#Converts the csv file into new format
for column in Parameters: # creates a for loop and checks each parameter column
    if Para_comp_check.loc[Para_comp_check['Parameter'] == column, 'check'].item() == 'Done':
        continue
    data_col = pd.DataFrame() #creates new dataframe for each element in Parameters list
    data_col = original_data.loc[original_data['param_name'] == column] #inputs only single element from parameters list into new dataframe: data_col
    data_col = data_col.drop(columns=['param_id','cams','param_name']) #drops unrequired columns from new dataframe: data_col
    data_col = data_col.rename(columns={'value':column, 'flag':(column+'_flag')}) #renames the required columns in the new dataframe: data_col
    data_col = data_col.sort_values(['site','epoch']) #sorts the dataframe based on site first and then epoch
    data_col = data_col.reset_index(drop=True) #resets the index for the new dataframe: data_col
    data_site_new = pd.DataFrame()
    print(column)
    for site_no in Site_List:
        data_site = pd.DataFrame()
        data_site = data_col.loc[data_col['site']==site_no] #creates new dataframe based on site for single element fromparameter
        data_site_columns = data_site.columns.values.tolist() #makes a list of the columns in dataframe: data_site
        no_time = [] #Empty list to check which times are not available for the parameter and site
        for time in Epoch: #for loop checks which times are not avaialble for the site
            if time not in list(data_site.epoch):
                no_time.append(time)
        data_site_empty = pd.DataFrame(columns=data_site_columns) #creates an empty dataframe based on columns in dataframe:data_sol
        data_site_empty['epoch'] = no_time #assigns no_time list to empty dataframe: data_site_empty
        data_site_empty['site'] = site_no #assigns site_no value to every element in site column of dataframe: data_site_empty
        data_site_empty['region'] = Location_df.loc[Location_df['site']==site_no, 'region'].iloc[0] #assigns region value to every element in site column of dataframe: data_site_empty based on site(from Location_df dataframe)
        data_site = pd.concat([data_site, data_site_empty], ignore_index=True)#combines dataframes data_site and data_site_empty
        data_site = data_site.sort_values(['epoch']) #sorts the dataframe based on epoch
        data_site = data_site.reset_index(drop=True) #resets the index for the dataframe: data_site
        data_site_new =  pd.concat([data_site_new, data_site], ignore_index=True) #combines data for all sites (one by one as per for loop) for one single parameter into single dataframe: data_site_new
        print(site_no)
    if i == 0: #for the first iteration of for loop, the data_site_new dataframe is saved in data_new dataframe
        data_new = data_site_new
    else: #for the remaining iterations of the for loop, the data_new dataframe combines/merges itslef with the data_site_new dataframes based on the site, region and epoch
        data_new = pd.concat([data_new,data_site_new], axis=1)
        data_new = pd.merge(data_new, data_site_new, how='outer', on=['site','epoch', 'region'])  
    print(i)
    i=i+1 # increases the iteration no
    Para_comp_check.check[Para_comp_check.Parameter == column] = 'Done' #assigns 'done' to column if for loop has been completed for it
    pickle.dump(i, open("iteration_no.p", "wb")) #checkpoint: saves iteration no i to be used later in case script fails
    pickle.dump(Para_comp_check, open("Parameters_converted.p", "wb")) #checkpoint: saves para_comp_check dataframe to be used later in case script fails
    pickle.dump(data_new, open("data_converted.p", "wb")) #checkpoint: saves data_new dataframe to be used later in case script fails
    data_new.to_csv('2013_New_format_sample.csv') # saves the data_new dataframe as a csv file in the same directory


