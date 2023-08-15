import os
import numpy as np
import pandas as pd

'''Get the data from the csv file'''
data_dir = os.path.join(os.getcwd(),'data')
data_name_list = os.listdir(data_dir)

full_data = pd.DataFrame(columns=['Hz','x1','x2','y1','y2'])
# Read the data from the csv file
for i in range(len(data_name_list)):
    data_name = data_name_list[i]
    # Get the data name without .csv
    hz = data_name.split('.xlsx')[0]
    print('Hz:',hz)
    data = pd.read_excel(os.path.join(data_dir,data_name))
    # Get some data from the data
    data = data[['X1','Y1','X2','Y2']]
    data = data.apply(lambda x: (x - 0) / (1300 - 0))
    # Rename some columns
    data = data.rename(columns={'X1':'x1','Y1':'x2','X2':'y1','Y2':'y2'})
    # Add a new column
    data['Hz'] = i#float(hz)
    data = data[['Hz','x1','x2','y1','y2']]
    # Save the data to the full_data
    full_data = pd.concat([full_data,data],ignore_index=True)
# Normalize the data
# norm_par = pd.DataFrame(columns=['mean','std'])
# for col in full_data.columns:
#     # Get the mean of the column
#     mean = np.mean(full_data[col])
#     # Get the std of the column
#     std = np.std(full_data[col])
#     # Normalize the column
#     full_data[col] = (full_data[col]-mean)/std
#     # Save the mean and std to the csv file
#     norm_par.loc[col] = [mean,std]
# # Save the norm_par to the csv file
# norm_par.to_csv('norm_par.csv',index=False)
# Save the full_data to the csv file
#full_data.to_csv('full_data.csv',index=False)