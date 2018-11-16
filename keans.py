import numpy as np
import matplotlib.pyplot as plt
import random

def productData(k):
    a = []
    for i in range(k):
        sampleNunber = 90
        s1 = random.random()*10
        s2 = random.random()*10
        mu = np.array([[s1, s2]])
        R = np.array([[1.0, 0], [0, 0.9]])
        s = mu + np.dot(np.random.randn(sampleNunber, 2), R)
        for j in s:
            b = []
            b.append(j[0])
            b.append(j[1])
            b.append(-1)
            a.append(b)
    return a

def kmeans(s, k):
    center = []
    h = {}
    for i in range(k):
        h[i] = []
        a = []
        a.append(random.random()*10)
        a.append(random.random()*10)
        center.append(a)
    for i in range(len(s)):
        num = findmin(s[i], center)
        h[num].append(s[i])
        s[i][2] = num
    center = updateCenter(h)
    number = 0
    while True:
        number = number+1
        h = {}
        for i in range(k):
            h[i] = []
        timesogchange = 0
        for i in s:
            num = findmin(i, center)
            h[num].append(i)
            if num != i[2]:
                i[2] = num
                timesogchange = timesogchange+1
        if timesogchange == 0:
            return h , number
        if number > 1000:
            return h , number
        center = updateCenter(h)

def findmin(x,center):
    b = -1
    h = 1000000
    for i in range(len(center)):
        a = (((center[i][0] - x[0]) ** 2 ) + ((center[i][1] - x[1]) ** 2 )) ** 0.5
        if a < h:
            h = a
            b = i;
    return b;


def updateCenter(h):
    center = []
    for i in h.keys():
        x = 0
        y = 0
        a = []
        for j in h[i]:
            x = x + j[0]
            y = y + j[1]
        if len(h[i]) != 0:
            a.append(x / len(h[i]))
            a.append(y / len(h[i]))
            center.append(a)
    return center

def paint(h):
    color = ['red','green','black','yellow','blue']
    for i in h.keys():
        x = []
        y = []
        for j in h[i]:
            x.append(j[0])
            y.append(j[1])
        plt.scatter(x,y,c=color[i])
    plt.show()


if __name__ == '__main__':
    k = 3
    s = productData(k)
    h, b = kmeans(s, k)
    print("迭代的次数为 ： {}".format(b))
    print(h)
    paint(h)
