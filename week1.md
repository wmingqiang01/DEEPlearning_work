# 什么是PyTorch？
PyTorch是一个python库，主要提供了两个功能：
1. GPU加速的张量计算
2. 构建在反向自动求导的深度神经网络

## 1.定义数据


```python
import torch

x = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
x
```




    tensor([[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]])




```python
y = torch.tensor(666)
y
```




    tensor(666)




```python
z = torch.tensor([1,2,3])
z
```




    tensor([1, 2, 3])



torch支持多种数据类型，创建时也有多种用法，ones，zeros，rand，normal等


```python
x = torch.empty(5,3)
x
```




    tensor([[-1.0836e-22,  9.0664e-43,  0.0000e+00],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00]])




```python
x = torch.rand(5,3)
x
```




    tensor([[0.3444, 0.3575, 0.2327],
            [0.2719, 0.2773, 0.8726],
            [0.1012, 0.2781, 0.8848],
            [0.0375, 0.4726, 0.6257],
            [0.0050, 0.4765, 0.4597]])




```python
x = torch.zeros(5,3,dtype=torch.long)
x
```




    tensor([[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]])




```python
y = x.new_ones(5,3)
y
```




    tensor([[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]])




```python
z = torch.randn_like(x,dtype=torch.float)
z
```




    tensor([[ 1.1557,  0.0308,  1.3028],
            [ 1.5709, -2.2808,  0.1252],
            [ 2.6811, -0.1219, -0.1229],
            [-1.0044, -0.9725,  0.0624],
            [ 0.8890, -0.2925, -0.4935]])



## 2.定义操作
用tensor进行各种运算，都是Function，包含
1. 基本运算，加减乘除、求幂求余
2. 布尔运算，
3. 线性运算、矩阵乘法，模、行列式等


```python
m =torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(m.size(0), m.size(1), m.size(), sep= ' -- ')
```

    3 -- 3 -- torch.Size([3, 3])



```python
m.numel()
```




    9




```python
m[:,1]
```




    tensor([2, 5, 8])




```python
m[0, :]
```




    tensor([1, 2, 3])




```python
v = torch.arange(1, 4)
v
```




    tensor([1, 2, 3])




```python
m @ v # 矩阵乘法
```




    tensor([14, 32, 50])




```python
m + torch.rand(3,3)
```




    tensor([[1.8799, 2.7695, 3.3915],
            [4.9748, 5.2245, 6.2632],
            [7.9083, 8.4652, 9.6025]])




```python
m.t()
```




    tensor([[1, 4, 7],
            [2, 5, 8],
            [3, 6, 9]])




```python
m.transpose(0, 1)
```




    tensor([[1, 4, 7],
            [2, 5, 8],
            [3, 6, 9]])




```python
torch.linspace(3, 8, 20)
```




    tensor([3.0000, 3.2632, 3.5263, 3.7895, 4.0526, 4.3158, 4.5789, 4.8421, 5.1053,
            5.3684, 5.6316, 5.8947, 6.1579, 6.4211, 6.6842, 6.9474, 7.2105, 7.4737,
            7.7368, 8.0000])




```python
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('fast')
```

