__author__ = 'multiangle'

# Matrix Factorization 的基本概念参考自 https://blog.csdn.net/lissanwen/article/details/51214275
#   但是好像在对 miu 的含义方面表述不是很清楚，也只是提到了对评分矩阵的估计方法和优化目标，
#   在对矩阵分解的具体方法方面含混不清

# 这边 (https://www.cnblogs.com/tbiiann/p/6535189.html) 自己实现了最简单版本的lfm (latent factor model)
#   这里我参考这篇文章做自己的lfm 方法，以此为基础加入对bias的处理， 时间变量的处理等

import numpy as np
import math
from matplotlib import pyplot as plt

def LFM_ed1(D, k, iter_times=1000, alpha=0.01, learn_rate=0.01):
    """
    此函数参考自 https://www.cnblogs.com/tbiiann/p/6535189.html
    :param D: 表示需要分解的评价矩阵, type = np.ndarray
    :param k: 分解的隐变量个数
    :param iter_times: 迭代次数
    :param alpha: 正则系数
    :param learn_rate: 学习速率
    :return: 分解得到的矩阵 U, V
    """

    pass

def LFM_ed2(D, k, iter_times=1000, alpha=0.01, learn_rate=0.01):
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

def LFM_ed3(D, k, iter_times=1000, alpha=0.01, learn_rate=0.01):
    """
    在 ed2 的基础上，加入了item 和 user的偏置， 以及整体的偏差 miu,
    不过ed3中的 bias 和 miu 都是常数，根据评分计算得到
    :param D: 表示需要分解的评价矩阵, type = np.ndarray
    :param k: 分解的隐变量个数
    :param iter_times: 迭代次数
    :param alpha: 正则系数
    :param learn_rate: 学习速率
    :return: U, V, B_u, B_v, miu, err_list
    """
    B_u = np.mean(D, axis=1)
    B_v = np.mean(D, axis=0)
    m, n = np.shape(D)
    s_bu = np.sum(B_u) * n
    s_bv = np.sum(B_v) * m
    miu = (np.sum(np.sum(D)) - s_bu - s_bv) / (m * n)

    U = np.random.rand(m, k)
    V = np.random.randn(k, n)
    B_u_expand = np.matmul(np.expand_dims(B_u, 1), np.ones([1, n]))
    B_v_expand = np.matmul(np.ones([m, 1]), np.expand_dims(B_v, 0))
    err_list = []
    for t in range(iter_times):
        D_est = np.matmul(U, V) + B_u_expand + B_v_expand + miu
        ERR = D - D_est
        U_grad = -2 * np.matmul(ERR, V.transpose()) + 2 * alpha * U
        V_grad = -2 * np.matmul(U.transpose(), ERR) + 2 * alpha * V
        U = U - learn_rate * U_grad
        V = V - learn_rate * V_grad

        ERR2 = np.multiply(ERR, ERR)
        ERR2_sum = np.sum(np.sum(ERR2))
        err_list.append(ERR2_sum)
    return U, V, B_u, B_v, miu, err_list

def LFM_ed4(D, k, iter_times=1000, alpha=0.01, learn_rate=0.01):
    """
    这里假设 bias 也是一个变量，需要迭代。
    :param D:
    :param k:
    :param iter_times:
    :param alpha:
    :param learn_rate:
    :return:
    """
    m, n = np.shape(D)
    miu = np.mean(np.mean(D))
    err_list = []

    B_u = np.random.rand(m)
    B_v = np.random.rand(n)
    U = np.random.rand(m, k)
    V = np.random.rand(k, n)
    for t in range(iter_times):
        B_u_expand = np.matmul(np.expand_dims(B_u, 1), np.ones([1, n]))
        B_v_expand = np.matmul(np.ones([m, 1]), np.expand_dims(B_v, 0))
        D_est = np.matmul(U, V) + B_u_expand + B_v_expand + miu
        ERR = D - D_est

        U_grad = -2 * np.matmul(ERR, V.transpose()) + 2 * alpha * U
        V_grad = -2 * np.matmul(U.transpose(), ERR) + 2 * alpha * V
        B_u_grad = -2 * np.sum(ERR, axis=1) + 2 * alpha * B_u
        B_v_grad = -2 * np.sum(ERR, axis=0) + 2 * alpha * B_v
        # B_u_grad = -2 * np.sum(ERR, axis=1)
        # B_v_grad = -2 * np.sum(ERR, axis=0)

        U = U - learn_rate * U_grad
        V = V - learn_rate * V_grad
        B_u = B_u - learn_rate * B_u_grad
        B_v = B_v - learn_rate * B_v_grad

        ERR2 = np.multiply(ERR, ERR)
        ERR2_sum = np.sum(np.sum(ERR2))
        err_list.append(ERR2_sum)

    return U, V, B_u, B_v, miu, err_list

if __name__=='__main__':
    D = np.array([[5,5,0,5],[5,0,3,4],[3,4,0,3],[0,0,5,3],[5,4,4,5],[5,4,5,5]])
    U, V, err_list = LFM_ed2(D, 3, iter_times=1000, learn_rate=0.01, alpha=0.01)
    # _, _,_, _, _, err_list = LFM_ed4(D, 3, iter_times=10000, learn_rate=0.01, alpha=0.01)
    print(err_list[-1])
    # err_log = np.log(np.array(err_list))
    plt.plot(err_list)
    plt.show()