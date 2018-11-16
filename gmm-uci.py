import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import multivariate_normal
import pandas as pd

times = 100  # 迭代次数

def loaddata(datafile):
    return np.array(pd.read_csv(datafile, sep=",", header=-1))

# 固定参数，进行Q的优化
def e_step(X, mu, cov, alpha):
    k = alpha.shape[0]
    number = X.shape[0]
    h = np.mat(np.zeros((number, k)))
    p = np.zeros((number, k))
    for i in range(k):
        p[:, i] = multivariate_normal(mean=mu[i], cov=cov[i]).pdf(X)
    p = np.mat(p)
    for i in range(k):
        h[:, i] = alpha[i] * p[:, i]
    for i in range(number):
        h[i, :] /= np.sum(h[i, :])
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
        alpha[k] = m / N
    cov = np.array(cov)
    return mu, cov, alpha
def gmm_em(X,k):
    '''
    :param X: 参数
    :param k: 类别个数
    :return: 混合高斯模型的参数
    '''
    # 预处理 对X 保持在0-1
    for i in range(X.shape[1]):
        max_ = X[:, i].max()
        min_ = X[:, i].min()
        X[:, i] = (X[:, i] - min_) / (max_ - min_)
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
    for i in range(100):
        h = e_step(X, mu, cov, alpha)
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
    k = 4
    s = loaddata("ucidata.txt")
    x = np.array(s)
    X = np.matrix(s)
    h = gmm_em(X, k)
