import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

acc_df = pd.read_csv('acc.csv')
maxdist_df = pd.read_csv('maxdist.csv')
fr_list = [14004, 15754.5, 1750.5, 17505, 22756.5, 28008, 29758.5, 31509, 3501, 35010, 7002, 8752.5]
# Plot acc and maxdist
for i in range(12):
    plt.figure()
    plt.plot(acc_df['0'], acc_df[str(i+1)], marker='o', label='Mean',c='b',markersize=5)
    plt.plot(maxdist_df['0'], maxdist_df[str(i+1)], marker='^', label='Max',c='g',markersize=5)
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('./fig/'+str(fr_list[i])+'Hz.png')
