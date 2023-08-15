import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the csv file
full_data = pd.read_csv('full_data.csv')
# Get the data from the full_data
check_data = full_data[full_data['Hz']==0]
#Find the data which is not in the range of [0,1]
check_data = check_data[(check_data['y1']<0)|(check_data['y1']>1)|(check_data['y2']<0)|(check_data['y2']>1)]
print(check_data)
# Hist the data
# check_data.hist()
# plt.show()