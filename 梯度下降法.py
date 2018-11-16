'''
@ by 刘磊 1163710108
  梯度下降法
    W的方向
    无正则项
    Wi+1 = Wi - alpha（学习率） * P(方向)
    P = X_T * X * Wi - X_T * Y
    有正则项
    Wi+1 = Wi - lamada（学习率） * P(方向)
    P = X_T * X * Wi - X_T * Y + lamada * W

    损失函数：   (WX - Y)^2
'''
import random
import numpy as np
import matplotlib.pyplot as plt
from math import *
# 拟合函数的前提条件
m = 15  # 最高维数
n = 10  # 数据个数
lamda = 0.001  # 惩罚系数
alpha = 0.0001  # 布长（学习率）
e = 0.0000001  # 误差范围

# 无惩罚项的梯度下降法
# 计算梯度向量的误差


def loss(Y, my_Y):
    loss = 0.0
    for i in range(len(Y)):
        loss += pow((my_Y[i] - Y[i]), 2)

    loss = 0.5 * loss * 1.0 / n
    return loss

def BatchGradientDescent(my_Y, Y, my_X, thata):
    my_copy = []
    for i in my_Y:
        my_copy.append(i)
    error = loss(Y, my_copy)
    X1 = np.array(my_X)

    while (1):
        new_thata = np.array(thata).reshape(len(thata), 1) - np.dot(alpha/n*np.eye(len(thata)), np.dot(np.dot(X1.T, X1),np.array(thata).reshape(len(thata), 1))) + np.dot(alpha/n*np.eye(len(thata)), np.dot(X1.T, np.array(Y).reshape(len(Y), 1)))
        thata = new_thata
        my_copy = np.array(np.dot(my_X, np.array(new_thata).reshape(len(thata), 1)).reshape(1, len(Y)))[0]
        new_error = loss(Y, my_copy)
        if abs(new_error - error) <= e:
            break
        error = new_error
    return thata
# 有惩罚项的梯度下降法
def lossReg(Y, my_Y, B):
    loss = 0.0
    b = np.array(B).reshape(1, m + 1)[0]
    for i in range(len(Y)):
        loss += pow((my_Y[i] - Y[i]), 2)
    for j in b:
        loss += lamda * j * j
    loss = 0.5 * loss * 1.0 / n
    return loss

def BatchGradientDescentReg(my_Y, Y, my_X, thata):
    my_copy = []
    for i in my_Y:
        my_copy.append(i)
    loss0 = lossReg(Y, my_copy, thata)
    X1 = np.array(my_X)

    while (1):
        # 计算下一个的thata 即W
        new_thata = np.array(thata).reshape(len(thata), 1) - np.dot(alpha / n * np.eye(len(thata)),
                                                                    np.dot(np.dot(X1.T, X1),
                                                                           np.array(thata).reshape(len(thata),
                                                                                                   1))) + np.dot(
            alpha / n * np.eye(len(thata)), np.dot(X1.T, np.array(Y).reshape(len(Y), 1))) + np.dot(alpha/n*np.eye(len(thata)), np.dot(lamda * np.eye(len(thata)),np.array(thata).reshape(len(thata),1)))

        thata = new_thata

        # 计算W
        my_copy = np.array(np.dot(my_X, np.array(new_thata).reshape(len(thata), 1)).reshape(1, len(Y)))[0]
        loss1 = lossReg(Y, my_copy, thata)
        if abs(loss1 - loss0) <= e:
            break
        loss0 = loss1
    return thata

if __name__ == "__main__":
    # 加入0均值的高斯噪声
    x = np.arange(0, 1, 1/n)
    y = np.sin(2*np.pi*x)
    for i in range(len(y)):
        y[i] = y[i] + random.gauss(0, 0.12)
    # 生成 范特蒙德行列式
    X = []
    for i in range(m+1):
        X.append(x ** i)
    X = np.mat(X).T
    #初始化w
    thata = []
    for i in range(0, m+1):
        thata.append([0])
    my_Y = np.array(np.dot(X, thata).reshape(1, len(y)))[0]
    thata = np.array(thata).reshape(1, len(thata))[0]
    W1 = BatchGradientDescent(my_Y, y, X, thata)
    W2 = BatchGradientDescentReg(my_Y, y, X, thata)
    #生成测试集
    x0 = np.arange(0, 0.9, 0.01)
    X2 = []
    for i in range(0, m+1):
        X2.append(x0 ** i)
    X0 = np.mat(X2).T
    result1 = np.dot(X0, np.array(W1).reshape(len(thata), 1))
    result2 = np.dot(X0, np.array(W2).reshape(len(thata), 1))
    # 显示图像输出结果
    print("无正则项 ： 多项式的系数为[w0 ..... w%d]" % (m))
    print(np.array(W1).reshape(1, len(W1)))
    print("有正则项 ： 多项式的系数为[w0 ..... w%d]" % (m))
    print(np.array(W2).reshape(1, len(W2)))
    # 计算回归曲线的代价
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
    plt.figure("梯度下降法")
    plt.title("梯度下降法")
    plt.scatter(x, y)
    plt.plot(x0, result1, 'r', label="$BGD$")
    plt.plot(x0, result2, 'b--', label="$BGD-REG$")
    plt.legend(loc=0)
    print("最高维数 :%d  数据个数 %d  惩罚系数中的lamada为：%f  学习率为： %f  误差为%f" % (m, n, lamda, alpha, e))
    plt.show()


