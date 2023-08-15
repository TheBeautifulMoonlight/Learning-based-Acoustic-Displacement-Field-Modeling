# Draw Loss Curve from loss.csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

loss_df = pd.read_csv('loss.csv')
# Add zero point
zero_df = pd.DataFrame([[0,0.5]], columns=['Epoch', 'Loss'])
loss_df = pd.concat([zero_df, loss_df], ignore_index=True)
plt.figure()
plt.plot(loss_df['Epoch'], loss_df['Loss'],c='b')

plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.show()