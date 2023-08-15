import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''构建神经网络模型'''
class MyNet(nn.Module):
    def __init__(self,n_input,n_output):
        super(MyNet, self).__init__()
        self.input = nn.Linear(n_input,16)
        self.hidden = nn.Linear(16,64)
        self.hidden2 = nn.Linear(64,16)
        self.out = nn.Linear(16,n_output)
    
    def forward(self,x):
        x = self.input(x)
        x = F.relu(x)
        x = self.hidden(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.out(x)
        return x

class Mydataset(Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        self.data = torch.from_numpy(data[:, :3]).float()
        self.target = torch.from_numpy(data[:, 3:]).float()

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y

    def __len__(self):
        return len(self.data)


def train(epoch, dataset, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Use device:', device)
    loss_rem = []
    acc_rem = []
    maxdist_rem = []
    model.to(device)
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for t in range(epoch):
        total_loss = 0
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            p = model(x)
            loss = loss_func(p, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if t % 100 == 0:
            accuracy, maxdist = acc(dataset, model)
            acc_rem.append([t]+accuracy)
            maxdist_rem.append([t]+maxdist)
            print('Epoch:', t, '| Loss:', total_loss/len(dataloader), '| Acc:', np.max(accuracy))
        if t % 10 == 0:
            loss_rem.append([t,total_loss/len(dataloader)])
        scheduler.step()
    # Plot the loss
    plt.figure()
    plt.plot([i[0] for i in loss_rem], [i[1] for i in loss_rem],marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.show()
    # save the model
    torch.save(model.state_dict(), 'model.pth')
    # save the loss to csv
    loss_df = pd.DataFrame(loss_rem, columns=['Epoch', 'Loss'])
    loss_df.to_csv('loss.csv', index=False)
    acc_df = pd.DataFrame(acc_rem)
    acc_df.to_csv('acc.csv', index=False)
    maxdist_df = pd.DataFrame(maxdist_rem)
    maxdist_df.to_csv('maxdist.csv', index=False)

def acc(dataset, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    predict_dist = np.zeros(12)
    max_predict_dist = np.zeros(12)
    min_predict_dist = np.ones(12)*100
    predict_num = np.zeros(12)
    for x, y in dataset:
        x = x.to(device)
        y = y.to(device)
        p = model(x)
        dis = torch.sqrt(torch.sum((p-y)**2))
        data_class = int(x[0])
        predict_dist[data_class] += dis.item()
        predict_num[data_class] += 1
        if dis.item() > max_predict_dist[data_class]:
            max_predict_dist[data_class] = dis.item()
        if dis.item() < min_predict_dist[data_class]:
            min_predict_dist[data_class] = dis.item()
    predict_dist = predict_dist/predict_num
    predict_dist = predict_dist.tolist()
    max_predict_dist = max_predict_dist.tolist()
    print('Min: ',min_predict_dist)
    return predict_dist, max_predict_dist

def test(dataset, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Use device:', device)
    model.to(device)
    loss_func = nn.MSELoss()
    total_loss = 0
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    accuracy,max_dist = acc(dataset, model)
    class_std = np.zeros(12)
    predict_num = np.zeros(12)
    for x, y in dataset:
        x = x.to(device)
        y = y.to(device)
        p = model(x)
        loss = loss_func(p, y)
        total_loss += loss.item()
        dis = torch.sqrt(torch.sum((p-y)**2))
        data_class = int(x[0])
        predict_num[data_class] += 1
        class_std[data_class] += (accuracy[data_class]-dis.item())**2
    class_std = class_std/predict_num
    print('Mse Loss:', total_loss/len(dataloader))
    print('Accuracy:', accuracy)
    print('Std:', class_std)
    print('Max: ',max_dist)
    '''Draw some data'''
    x, y = dataset[0:100]
    x = x.to(device)
    y = y.to(device)
    p = model(x)
    x = x.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    p = p.cpu().detach().numpy()
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x[:,-2], x[:,-1], y[:,1], c='r', label='Input')
    ax.scatter3D(x[:,-2], x[:,-1], p[:,1], c='b', label='Predict')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.legend()
    plt.show()
    return total_loss/len(dataloader)
        
data_name = './Data/full_data.csv'
data_df = pd.read_csv(data_name)
data = data_df.values
dataset = Mydataset(data)
print('Len of dataset:', len(dataset))
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
model = MyNet(3,2)
'''Test'''
before_loss = test(dataset, model)
'''Train'''
train(3000, dataset, model)
model.load_state_dict(torch.load('model.pth'))
'''Test'''
model.eval()
after_loss = test(dataset, model)
print('Loss improve:', before_loss-after_loss)