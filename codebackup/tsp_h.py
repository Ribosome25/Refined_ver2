# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 10:31:16 2020

@author: Ruibo
"""

'''
连续型——Hopfield神经网络求解TSP
1、初始化权值（A,D,U0）
2、计算N个城市的距离矩阵dxy
3、初始化神经网络的输入电压Uxi和输出电压Vxi
4、利用动力微分方程计算：dUxi/dt
5、由一阶欧拉方法更新计算：Uxi(t+1) = Uxi(t) + dUxi/dt * step
6、由非线性函数sigmoid更新计算：Vxi(t) = 0.5 * (1 + th(Uxi/U0))
7、计算能量函数E
8、检查路径是否合法
'''
import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils.graph_shortest_path import graph_shortest_path

class Hopfield:
    def __init__(self, U0 = 0.0009, step = 0.0001, num_iter = 10000):
        self.U0 = U0
        self.step = step
        self.num_iter = num_iter
    
    def calc_U(self, U, du, step):
        return U + du * step
    
    def calc_V(self, U, U0):
        return 1 / 2 * (1 + np.tanh(U / U0))
    
    def calc_du(self, V, distance):
        N = self.N
        a = np.sum(V, axis=0) - 1  # 按列相加
        b = np.sum(V, axis=1) - 1  # 按行相加
        t1 = np.zeros((N, N))
        t2 = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                t1[i, j] = a[j]
        for i in range(N):
            for j in range(N):
                t2[j, i] = b[j]
        # 将第一列移动到最后一列
        c_1 = V[:, 1:N]
        c_0 = np.zeros((N, 1))
        c_0[:, 0] = V[:, 0]
        c = np.concatenate((c_1, c_0), axis=1)
        c = np.dot(distance, c)
        return -A * (t1 + t2) - D * c
        
    def calc_energy(self, V, distance):
        t1 = np.sum(np.power(np.sum(V, axis=0) - 1, 2))
        t2 = np.sum(np.power(np.sum(V, axis=1) - 1, 2))
        idx = [i for i in range(1, N)]
        idx = idx + [0]
        Vt = V[:, idx]
        t3 = distance * Vt
        t3 = np.sum(np.sum(np.multiply(V, t3)))
        e = 0.5 * (A * (t1 + t2) + D * t3)
        return e
        
    def _initialize(self, distance):
        N = distance.shape[0]
        self.A = N * N
        self.D = N / 2
        self.U = 1 / 2 * U0 * np.log(N - 1) + (2 * (np.random.random((N, N))) - 1)
        self.V = calc_V(U, U0)
        self.energys = np.array([0.0 for x in range(num_iter)])  # 每次迭代的能量
        self.best_distance = np.inf  # 最优距离
        self.best_route = []  # 最优路线
        self.H_path = []  # 哈密顿回路
        
    def fit(self, distance):
        self._initialize(distance)
        distance = graph_shortest_path(distance)
        np.fill_diagonal(distance, np.inf)
        self.distance = distance
        
        for n in range(num_iter):
            du = self.calc_du(self.V, distance)
            U = self.calc_U(self.U, du, step)
            V = calc_V(U, self.U0)
            self.energys[n] = self.calc_energy(V, distance)