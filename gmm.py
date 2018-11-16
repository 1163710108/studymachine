import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import multivariate_normal

times =  50 # 迭代次数


def productData(k):
    '''
    :param k: k类
    :return:
    '''
    a = []
    r = []
    for i in range(k):
        sampleNunber = 130
        s1 = random.random()*10
        s2 = random.random()*20
        r.append([s1, s2])
        mu = np.array([[s1, s2]])
        R = np.array([[1.0, 0], [0, 0.9]])
        s = mu + np.dot(np.random.randn(sampleNunber, 2), R)
        for j in s:
            b = []
            b.append(j[0])
            b.append(j[1])
            a.append(b)
    r= np.array(r)
    print("生成数据的原始参数为")
    print("均值为")
    print(r)
    print("方差为")
    print(R)
    return a

# 固定参数，进行Q的优化
def e_step(X, mu, cov, alpha):
    k = alpha.shape[0]
    number = X.shape[0]
    h = np.mat(np.zeros((number, k)))
    p = np.zeros((number, k))
    for i in range(k):
        p[:, i] = multivariate_normal(mean=mu[i], cov=cov[i]).pdf(X) # 均值为mu[i] he cov[i]  的X
    p = np.mat(p)
    for i in range(k):
        h[:, i] = alpha[i] * p[:, i]   # 计算h
    for i in range(number):
        h[i, :] /= np.sum(h[i, :])   #h的总和
    return h;

# 固定Q进行参数的优化
def m_step(X,h):
    N, D = X.shape
    k = h.shape[1]
    # 优化参数
    mu = np.zeros((k, D))
    cov = []
    alpha = np.zeros(k)
    for k in range(k):
        m = np.sum(h[:, k])
        for d in range(D):
            mu[k, d] = np.sum(np.multiply(h[:, k], X[:, d])) / m
        cov_k = np.mat(np.zeros((D, D)))
        for i in range(N):
            cov_k += h[i, k] * (X[i] - mu[k]).T * (X[i] - mu[k]) / m
        cov.append(cov_k)
        # 计算先验概率 Nk/N
        alpha[k] = m / N
    cov = np.array(cov)
    return mu, cov, alpha
def gmm_em(X,k):
    '''
    :param X: 参数
    :param k: 类别个数
    :return: 混合高斯模型的参数
    '''
    D = X.shape[1]
    mu = np.random.rand(k, D)  # 初始均值
    cov = np.array([np.eye(D)] * k)  # 初始的方差
    alpha = np.array([1.0 / k] * k)  # 初始的时候每个模型的概率 1/k
    print("初始化为")
    print("mu 均值")
    print(mu)
    print("cov 方差")
    print(cov)
    print("alpha 概率")
    print(alpha)
    # 迭代
    for i in range(100):
        # e步 - 分类
        h = e_step(X, mu, cov, alpha)
        # m步 - 参数确定
        mu, cov, alpha = m_step(X, h)
    print("迭代之后")
    print("mu 均值")
    print(mu)
    print("cov 方差")
    print(cov)
    print("alpha 概率")
    print(alpha)
    return h


if __name__ == '__main__':
    k = 3
    s = productData(k)
    x = np.array(s)
    X = np.matrix(s)
    h = gmm_em(X, k)
    category = []
    y = np.array(h)
    for i in range(len(y)):
        b = 0
        index = 0
        for j in range(len(y[i])):
            if b <= y[i][j]:
                b = y[i][j]
                index = j
        category.append(index)
    color = ['red', 'green', 'black', 'yellow', 'blue']
    for i in range(len(category)):
        plt.scatter(x[i][0],x[i][1],c=color[category[i]])
    plt.show()

