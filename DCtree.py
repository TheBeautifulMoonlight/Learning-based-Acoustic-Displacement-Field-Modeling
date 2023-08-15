import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

# Read the data from the csv file
full_data = pd.read_csv('full_data.csv')
# Get the data from the full_data
X = full_data[['X1','X2','Hz']]
Y = full_data[['Y1','Y2']]
# Split the data to train and test
train_num = int(len(X)*0.6)
X_train = X[:train_num]
Y_train = Y[:train_num]
X_test = X[train_num:]
Y_test = Y[train_num:]
# Create a regression tree
reg_tree = tree.DecisionTreeRegressor(max_depth=10)
# Train the regression tree
reg_tree.fit(X,Y)
# Print the score of the regression tree
print('Train Score:',reg_tree.score(X,Y))
# Predict the Y
Y_pred = reg_tree.predict(X_test)
# Print the score of the regression tree
print('Test Score:',reg_tree.score(X_test,Y_test))
# Show the tree
# plt.figure(figsize=(20,20))
# tree.plot_tree(reg_tree,filled=True)
# plt.show()