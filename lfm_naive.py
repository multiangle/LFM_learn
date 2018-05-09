__author__ = 'multiangle'

# Matrix Factorization 的基本概念参考自 https://blog.csdn.net/lissanwen/article/details/51214275
#   但是好像在对 miu 的含义方面表述不是很清楚，也只是提到了对评分矩阵的估计方法和优化目标，
#   在对矩阵分解的具体方法方面含混不清

# 这边 (https://www.cnblogs.com/tbiiann/p/6535189.html) 自己实现了最简单版本的lfm (latent factor model)
#   这里我参考这篇文章做自己的lfm 方法，以此为基础加入对bias的处理， 时间变量的处理等

import numpy as np
import math
from matplotlib import pyplot as plt

def LFM_naive(D, k, iter_times=1000, alpha=0.01, learn_rate=0.01):
    '''
    此函数实现的是最简单的 LFM 功能
    :param D: 表示需要分解的评价矩阵, type = np.ndarray
    :param k: 分解的隐变量个数
    :param iter_times: 迭代次数
    :param alpha: 正则系数
    :param learn_rate: 学习速率
    :return:  分解完毕的矩阵 U, V, 以及误差列表err_list
    '''
    assert type(D) == np.ndarray
    m, n = D.shape  # D size = m * n
    U = np.random.rand(m, k)    # 为何要一个均匀分布一个正态分布？
    V = np.random.randn(k, n)
    err_list = []
    for t in range(iter_times):
        # 这里，对原文中公式推导我认为是推导正确的，但是循环效率太低了，可以以矩阵形式计算
        D_est = np.matmul(U, V)
        ERR = D - D_est
        U_grad = -2 * np.matmul(ERR, V.transpose()) + 2 * alpha * U
        V_grad = -2 * np.matmul(U.transpose(), ERR) + 2 * alpha * V
        U = U - learn_rate * U_grad
        V = V - learn_rate * V_grad

        ERR2 = np.multiply(ERR, ERR)
        ERR2_sum = np.sum(np.sum(ERR2))
        err_list.append(ERR2_sum)
    return U, V, err_list
    

if __name__=='__main__':
    D = np.array([[5,5,0,5],[5,0,3,4],[3,4,0,3],[0,0,5,3],[5,4,4,5],[5,4,5,5]])
    U, V, err_list = LFM_naive(D, 3, iter_times=100, learn_rate=0.01, alpha=0.01)
    print(err_list[-1])
    err_log = np.log(np.array(err_list))
    plt.plot(err_list)
    plt.show()