'''
@ by 刘磊 1163710108
    最小二乘法无正则项
    W=(X^T*X + lamada*I)^-1  * X^T *Y;
    (X^T*X + lamada*I)*W = X^T *Y;
'''
import matplotlib.pyplot as plt
import numpy as np
import random
import math
# 拟合函数的前提条件
#最高维数
m = 15
#数据个数
n = 10
# 惩罚系数
lamada = 0.001
t = m+1
# 计算损失
def loss(Y, my_Y):
    loss = 0.0
    for i in range(len(Y)):
        loss += pow((my_Y[i] - Y[i]), 2)

    loss = 0.5 * loss * 1.0 / n
    return loss
# 生成数据，0均值的高斯函数，作为噪声！
x = np.arange(0, 1, 1/n)
y = np.sin(2*np.pi*x)
for i in range(len(x)):
    y[i] = y[i] + random.gauss(0, 0.12)
Y = np.array(y).reshape((len(y), 1))
# 求结过程
#求解范德蒙行列式
X1 = []
for i in range(0, t):
    X1.append(x ** i)
X = np.mat(X1).T
#转制矩阵
X_T = X.T
#带入公式
W = np.dot(np.dot(np.dot(X_T, X).I, X_T), Y)
W1 = np.dot(np.dot((np.dot(X_T, X) + lamada*np.eye(t)).I, X_T), Y)
#生成测试集
x0 =  np.arange(0,0.9,0.01)
X2 = []
for i in range(0,t):
    X2.append(x0 ** i)
X0 = np.mat(X2).T
result1 = np.dot(X0, W)
result2 = np.dot(X0, W1)
print("无正则项 ： 多项式的系数为[w0 ..... w%d]" % (m ))
print(np.array(W).reshape(1, len(W)))
print("有正则项 ： 多项式的系数为[w0 ..... w%d]" % (m ))
print(np.array(W1).reshape(1, len(W1)))
# 计算误差
y0 = np.sin(2 * np.pi * x0)
my_1 = np.array(np.dot(X, W)).reshape(1, n)[0]
my_2 = np.array(np.dot(X, W1)).reshape(1, n)[0]
my_3 = np.array(result1).reshape(1, 90)[0]
my_4 = np.array(result2).reshape(1, 90)[0]
print("无正则项目的训练集代价 ： %f" % loss(np.array(y).reshape(1,n)[0],my_1))
print("有正则项目的训练集代价 ： %f" % loss(np.array(y).reshape(1,n)[0],my_2))
print("无正则项目的测试集代价 ： %f" % loss(np.array(y0).reshape(1, 90)[0], my_3))
print("有正则项目的测试集代价 ： %f" % loss(np.array(y0).reshape(1, 90)[0], my_4))

# 显示图像
plt.figure("最小二乘法")
plt.title("最小二乘法")
plt.xlabel('x ')
plt.ylabel('y ')
plt.scatter(x, y)
plt.plot(x0, result1, 'r', label="$LS$")
plt.plot(x0, result2, 'b--', label="$LS-REG$")
plt.legend(loc=0)
print("最高维数 :%d  数据个数 %d  惩罚系数中的lamada为：%f "%(m, n, lamada))
plt.show()

