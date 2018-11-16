'''
    E = (X^T*X + lamada*I)*W - X^T *Y;
    求其最小值

    共轭梯度法 : W前的矩阵不为对称正定的方阵
    详细情况见实验报告
'''

import random
import numpy as np
import matplotlib.pyplot as plt
from math import *

# 拟合函数的前提条件
m = 9  # 最高维数
n = 10  # 数据个数
lamda = 0.001  # 惩罚系数
e = 0.0000001  # 误差范围


# 计算回归曲线的代价
def loss(Y, my_Y):
    loss = 0.0
    for i in range(len(Y)):
        loss += pow((my_Y[i] - Y[i]), 2)

    loss = 0.5 * loss * 1.0 / n
    return loss
# 无惩罚项目的共轭梯度法

def gonetidu(X, Y, B):
    '''
    #稳定双共轭梯度下降
    '''
    A = np.dot(X.T, X)
    b = np.dot(X.T, Y)
    r0 = b - np.dot(A, B)
    d0 = b - np.dot(A, B)
    for i in range(n):
        r1 = np.array(r0).reshape(1, n)[0]
        al = (np.array(np.dot(r0.T, r0))[0][0])/(np.array(np.dot(np.dot(d0.T, A), d0))[0][0])
        B = B + np.dot(al*np.eye(n), np.array(d0).reshape(n, 1))
        r0 = r0 - np.dot(np.dot(al*np.eye(n), A), d0)
        h = np.array(r0).reshape(1, n)[0]
        tatl = 0.0
        for j in h:
            tatl += j ** 2
        tatl0 = pow(tatl, 0.5)
        if tatl0 <= e or i+1 == n:
            break
        tatl2 = 0.0
        for j in r1:
            tatl2 += j ** 2
        beta = tatl / tatl2
        d0 = r0 + np.dot(beta*np.eye(n), d0)
    return B


# 有惩罚项的共轭梯度法

def gonetiduReg(X, Y, B):
    '''
           #稳定双共轭梯度下降
           '''
    A = np.dot(X.T, X) + lamda*np.eye(n)
    b = np.dot(X.T, Y)
    r0 = b - np.dot(A, B)
    d0 = b - np.dot(A, B)
    for i in range(n):
        r1 = np.array(r0).reshape(1, n)[0]
        al = (np.array(np.dot(r0.T, r0))[0][0]) / (np.array(np.dot(np.dot(d0.T, A), d0))[0][0])
        B = B + np.dot(al * np.eye(n), np.array(d0).reshape(n, 1))
        r0 = r0 - np.dot(np.dot(al * np.eye(n), A), d0)
        h = np.array(r0).reshape(1, n)[0]
        tatl = 0.0
        for j in h:
            tatl += j ** 2
        tatl0 = pow(tatl, 0.5)
        if tatl0 <= e or i + 1 == n:
            break
        tatl2 = 0.0
        for j in r1:
            tatl2 += j ** 2
        beta = tatl / tatl2
        d0 = r0 + np.dot(beta * np.eye(n), d0)
    return  B


if m + 1 == n:
    # 生成数据，产生噪声
    x = np.arange(0, 1, 1 / n)
    y = np.sin(2 * np.pi * x)
    for i in range(len(y)):
        y[i] = y[i] + random.gauss(0, 0.12)
    # 生成 范特德蒙行列式
    length = len(y)
    X = []
    for i in range(m + 1):
        X.append(x ** i)
    X = np.mat(X).T
    # Y 矩阵
    Y = np.array(y).reshape(length, 1)
    # 初始W
    thata = []
    for i in range(0, m + 1):
        thata.append([0])
    thata = np.array(thata).reshape(length, 1)
    # 共轭梯度求解
    W1 = gonetidu(X, Y, thata)
    W2 = gonetiduReg(X, Y,thata)
    # 生成测试集
    # 生成测试集
    x0 = np.arange(0, 0.9, 0.01)
    X2 = []
    for i in range(0, n):
        X2.append(x0 ** i)
    X0 = np.mat(X2).T
    result1 = np.dot(X0, W1)
    result2 = np.dot(X0, W2)
    print("无正则项 ： 多项式的系数为[w0 ..... w%d]" % (m))
    print(np.array(W1).reshape(1, len(W1)))
    print("有正则项 ： 多项式的系数为[w0 ..... w%d]" % (m))
    print(np.array(W2).reshape(1, len(W2)))
    # 计算误差
    y0 = np.sin(2 * np.pi * x0)
    my_1 = np.array(np.dot(X, W1)).reshape(1, n)[0]
    my_2 = np.array(np.dot(X, W2)).reshape(1, n)[0]
    my_3 = np.array(result1).reshape(1, 90)[0]
    my_4 = np.array(result2).reshape(1, 90)[0]
    print("无正则项目的训练集代价 ： %f" % loss(np.array(y).reshape(1, n)[0], my_1))
    print("有正则项目的训练集代价 ： %f" % loss(np.array(y).reshape(1, n)[0], my_2))
    print("无正则项目的测试集代价 ： %f" % loss(np.array(y0).reshape(1, 90)[0], my_3))
    print("有正则项目的测试集代价 ： %f" % loss(np.array(y0).reshape(1, 90)[0], my_4))

    # 显示图像
    plt.scatter(x, y)
    plt.plot(x0, result1, 'r', label="$CG$")
    plt.plot(x0, result2, 'b--', label="$CG-REG$")
    plt.legend(loc=0)
    plt.show()

else:
    print("范德蒙行列式不是方阵，无法使用共轭矩阵")