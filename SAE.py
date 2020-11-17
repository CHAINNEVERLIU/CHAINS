import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error  #均方误差
from sklearn.metrics import r2_score            #R square
import torch.nn.functional as F
from torch.autograd import Variable
import math, copy, time
import random

""""
标准的脱丁烷塔数据集作为测试数据集验证程序
"""
data = pd.read_table('D:\\科研相关\\OASAE（matlab）\\DeepLearnToolbox-SAE\\data\\debutanizer_data.txt', sep='\s+', header=None)
data = data.values  #[2394, 8]
x_temp = data[:, :7]
y_temp = data[:, 7]

#动态拓展
x_new = np.zeros([2390, 13])
x_6 = x_temp[:, 4]
x_9 = (x_temp[:, 5]+x_temp[:, 6])/2
x_new[:, :5] = x_temp[4:2394, :5]
x_new[:, 5] = x_6[3:2393]
x_new[:, 6] = x_6[2:2392]
x_new[:, 7] = x_6[1:2391]
x_new[:, 8] = x_9[4:2394]
x_new[:, 9] = y_temp[3:2393]
x_new[:, 10] = y_temp[2:2392]
x_new[:, 11] = y_temp[1:2391]
x_new[:, 12] = y_temp[:2390]
y_new = y_temp[4:2394]
y_new = y_new.reshape([-1, 1])

#划分数据集
x_new = torch.from_numpy(x_new).float()
y_new = torch.from_numpy(y_new).float()
train_x = x_new[:1000, :]
train_y = y_new[:1000]

x_validation = x_new[1000:1600, :]
y_validation = y_new[1000:1600]

test_x = x_new[1600:2390, :]
test_y = y_new[1600:2390]

uns_batchsize = 50
batchsize = 20

torch_train_dataset = Data.TensorDataset(train_x, train_y)
uns_trainloader = Data.DataLoader(dataset=torch_train_dataset, batch_size=uns_batchsize, shuffle=False, num_workers=0, pin_memory=False)
train_loader = Data.DataLoader(dataset=torch_train_dataset, batch_size=batchsize, shuffle=False, num_workers=0, pin_memory=False)



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(0)


class AutoEncoder(nn.Module):
    """
    标准的AE构建
    """
    def __init__(self, inputsize, hiddensize):
        super(AutoEncoder, self).__init__()
        self.inputsize = inputsize
        self.hiddensize = hiddensize
        self.encoder = nn.Linear(inputsize, hiddensize, bias=True)
        self.decoder = nn.Linear(hiddensize, inputsize, bias=True)
        self.activationfunction = torch.sigmoid

    def forward(self, input, rep=False):
        """"
        repeat参数用来指定此AE模块是否是输出模块
        """
        hidden = self.activationfunction(self.encoder(input))      #input layer-->hidden layer
        if rep is False:
            return self.activationfunction(self.decoder(hidden))
        else:
            return hidden


class StackedAE(nn.Module):
    """"
    标准的堆叠自编码器构建
    encoder_list 输入的是一个AE网络架构列表
    """
    def __init__(self, size):
        super(StackedAE, self).__init__()
        self.n = len(size)
        self.sae = []
        for i in range(1, self.n):
            self.sae.append(AutoEncoder(size[i-1], size[i]))
        self.proj = nn.Linear(size[self.n - 1], 1)

    def forward(self, x, k, unsp=True):
        """"
        k :代表的是无监督预训练第几个AE
        unsp: 代表模型进行的是无监督预训练还是有监督微调
        """
        out = x
        if unsp is True:
            if k == 0 :
                return out, self.sae[k](out)

            else:
                for i in range(k):
                    for param in self.sae[i].parameters():  # 冻结参数
                        param.requires_grad = False

                    out = self.sae[i](out, rep=True)

                inputs = out
                out = self.sae[k](out)
                return inputs, out

        else:
            for i in range(self.n-1):
                for param in self.sae[i].parameters():  # 冻结参数
                    param.requires_grad = True

                out = self.sae[i](out, rep=True)
            out = torch.sigmoid(self.proj(out))
            return out

def trainAE(model, trainloader, epochs, trainlayer):
    """"
    训练单个AE的代码
    """
    optimizer = torch.optim.Adam(model.sae[trainlayer].parameters(), lr=0.03)
    loss_func = nn.MSELoss()
    for j in range(epochs):
        sum_loss = 0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            input, outs = model(inputs, trainlayer, unsp=True)
            loss = loss_func(input, outs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.detach().item()
            print('AE', trainlayer, '|unsupervised train AE epoch:', j, '|batch:', i, '|loss:', loss.data.numpy())

    return model

def trainSAE(SAEmodel, trainloader, un_trainloader, unepochs, epochs, numAE):
    """"
    训练SAE网络
    """
    #step 1: optimizer 定义
    #optimizer = torch.optim.Adam(SAEmodel.parameters(), lr=0.03)   #注意这里要把每个AE的参数都添加进去
    optimizer = torch.optim.Adam(
        [
            {'params': SAEmodel.parameters(), 'lr': 0.035},
            {'params': SAEmodel.sae[0].parameters(), 'lr': 0.01},
            {'params': SAEmodel.sae[1].parameters(), 'lr': 0.01},
            {'params': SAEmodel.sae[2].parameters(), 'lr': 0.01}
        ]
    )
    loss_func = nn.MSELoss()

    #step 2: train AE
    for i in range(numAE):
       SAEmodel = trainAE(SAEmodel, un_trainloader, unepochs, i)

    #step 3: train SAE
    Loss = []
    for i in range(epochs):
        sum_loss = 0
        for j, data in enumerate(trainloader):
            inputs, labels = data
            pre = SAEmodel(inputs, j, unsp=False)
            loss = loss_func(pre, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.detach().item()
            print('SAE epoch:', i, '|batch:', j, '|loss:', loss.data.numpy())
        Loss.append(sum_loss)
    plt.figure()
    plt.plot(range(len(Loss)), Loss, color='b')
    plt.show()
    return SAEmodel

struct = [13, 10, 7, 5]
num_AE = len(struct)-1
model = StackedAE(struct)

model = trainSAE(model, train_loader, uns_trainloader, unepochs=300, epochs=100, numAE=num_AE)

#训练集表现
pre = model(train_x, 0, unsp=False)
output_train = pre.data.numpy()
plt.figure()
plt.plot(range(len(output_train)), output_train, color='b', label='y_trainpre')
plt.plot(range(len(output_train)), train_y.data.numpy(), color='r', label='y_true')
plt.legend()
plt.show()
train_rmse = np.sqrt(mean_squared_error(output_train, train_y.data.numpy()))
train_r2 = r2_score(output_train, train_y.data.numpy())
print('train_rmse = ' + str(round(train_rmse, 5)))
print('r2 = ', str(train_r2))

# 测试集表现
outs_test = model(test_x, 0, unsp=False)
output_test = outs_test.data.numpy()
plt.figure()
plt.plot(range(len(output_test)), output_test, color='b', label='y_testpre')
plt.plot(range(len(output_test)), test_y.data.numpy(), color='r', label='y_true')
plt.legend()
plt.show()
test_rmse = np.sqrt(mean_squared_error(output_test, test_y.data.numpy()))
test_r2 = r2_score(output_test, test_y.data.numpy())
print('test_rmse = ' + str(round(test_rmse, 5)))
print('r2 = ', str(test_r2))




