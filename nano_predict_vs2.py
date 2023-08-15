import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import copy
import numpy as np
import os
import csv

'''构建神经网络模型'''
class MyNet(nn.Module):
    def __init__(self,n_input,n_output):
        super(MyNet, self).__init__()
        self.input = nn.Linear(n_input,64)
        self.liner1 = nn.Linear(64,64)
        self.liner2 = nn.Linear(64,64)
        self.out = nn.Linear(64,n_output)
    
    def forward(self,x):
        x = self.input(x)
        x = F.relu(x)
        x = self.liner1(x)
        x = self.liner2(x)
        x = F.relu(x)
        x = self.out(x)
        return x

def train_model(epoch, train_dataLoader, test_dataLoader,model,loss_func,optimizer):
    # 训练模型
    best_model = None
    train_loss = 0
    test_loss = 0
    best_loss = 1000
    epoch_cnt = 0
    for e in tqdm(range(epoch)):
        total_train_loss = 0
        total_train_num = 0
        total_test_loss = 0
        total_test_num = 0
        for x, y in train_dataLoader:
            x_num = len(x)
            p = model(x)
            loss = loss_func(p, y)
            # print('P:',p)
            # print('Y:',y)
            # print('loss:',loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_train_num += x_num
        train_loss = total_train_loss / total_train_num
        for x, y in test_dataLoader:
            x_num = len(x)
            p = model(x)
            loss = loss_func(p, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_test_loss += loss.item()
            total_test_num += x_num
        test_loss = total_test_loss / total_test_num
        
        # early stop
        if best_loss > test_loss:
            best_loss = test_loss
            best_model = copy.copy(model)
            epoch_cnt = 0
        else:
            epoch_cnt += 1
    #torch.save(best_model.state_dict(), 'data/nano.pth')

predict_name = 'Y2-Y1'
data = pd.read_excel('Test Data2.xlsx')
#data = data.apply(lambda x: (x - min(x)) / (max(x) - min(x))) # 数据归一化
data_x1 = data['X1']
data_x2 = data['Y1']
data_y = data[predict_name]
'''初始数据可视化'''
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data_x1, data_x2, data_y)

ax.set_zlabel(predict_name, fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
plt.show()

'''设置学习参数'''
model = MyNet(2,1)
learning_rate = 0.001
epoch_num = 500
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
loss = nn.MSELoss()

'''处理训练数据转换为tensor dataset'''
train_x = [[data_x1[i],data_x2[i]] for i in range(len(data))][:80]
train_y = [[data_y[i]] for i in range(len(data))][:80]
train_x = torch.tensor(train_x).float()
train_y = torch.tensor(train_y).float()
test_x = [[data_x1[i],data_x2[i]] for i in range(len(data))][80:]
test_y = [[data_y[i]] for i in range(len(data))][80:]
test_x = torch.tensor(test_x).float()
test_y = torch.tensor(test_y).float()
# print('Train X:',train_x)
# print('Train Y:',train_y)
train_data = TensorDataset(train_x, train_y)
train_dataLoader = DataLoader(train_data, batch_size=20)
test_data = TensorDataset(test_x, test_y)
test_dataLoader = DataLoader(test_data, batch_size=20)

'''开始学习'''
train_model(epoch_num,train_dataLoader,test_dataLoader,model,loss,optimizer)
'''测试模型'''
#model.load_state_dict(torch.load('data/nano.pth'))
predict = model(test_x)
prediction = predict.tolist()
'''训练结果可视化'''
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data_x1[80:], data_x2[80:], data_y[80:],c='blue')
ax.scatter(data_x1[80:], data_x2[80:], prediction,c='red')
ax.set_zlabel(predict_name, fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
'''保存预测数据'''
csv_file = open('PredictCompare'+predict_name+'.csv', 'w', newline='', encoding='gbk')
writer = csv.writer(csv_file)
draw_x1 = list(data_x1[80:])
draw_x2 = list(data_x2[80:])
draw_x1.insert(0,'x')
writer.writerow(draw_x1)
draw_x2.insert(0,'y')
writer.writerow(draw_x2)
predict_list = []
for i in range(len(prediction)):
    predict_list.append(prediction[i][0])
predict_list.insert(0,predict_name)
writer.writerow(predict_list)
plt.show()