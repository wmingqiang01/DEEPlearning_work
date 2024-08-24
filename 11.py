# import
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from IPython import display
from plot_lib import plot_data, plot_model, set_default


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
seed = 12345
random.seed(seed)
torch.manual_seed(seed)

N = 1000  # 每类样本的数量
D = 2  # 每类样本的特征维度
C = 3  # 样本类别
H = 100  # 隐藏层单元数量
# %%
X = torch.zeros(N * C, D).to(device)
Y = torch.zeros(N * C, dtype=torch.long).to(device)
for c in range(C):
    index = 0
    t = torch.linspace(0, 1, N)
    inner_var = torch.linspace((2 * math.pi / C) * c, (2 * math.pi / C) * (c + 2), N)
    for ix in range(N * c, N * (c + 1)):
        X[ix] = t[index] * torch.FloatTensor((math.sin(inner_var[index]), math.cos(inner_var[index])))
        Y[ix] = c
        index += 1
print('Shapes：')
print(X.size())
print(Y.size())
# %%
plot_data(X, Y)
# %% md
## 1.构建线性分类模型
# %%
lr = 1e-3
lambda_12 = 1e-5

# nn用于创建模型，每个模型包含weight和bias
model = torch.nn.Sequential(
    nn.Linear(D, H),
    nn.Linear(H, C)
)

# loss，本处使用交叉熵函数
critierion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=lambda_12)

# 训练
for epoch in range(1000):
    y_pred = model(X)
    loss = critierion(y_pred, Y)
    score, predicted = torch.max(y_pred, 1)
    acc = (predicted == Y).sum().float() / Y.size(0)
    print('[EPROCH]:%i, [LOSS]:%.3f, [ACC]:%.3f' % (epoch, loss.item(), acc))
    display.clear_output(wait=True)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()