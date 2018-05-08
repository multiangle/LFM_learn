__author__ = 'multiangle'

# Matrix Factorization 的基本概念参考自 https://blog.csdn.net/lissanwen/article/details/51214275
#   但是好像在对 miu 的含义方面表述不是很清楚，也只是提到了对评分矩阵的估计方法和优化目标，
#   在对矩阵分解的具体方法方面含混不清

# 这边 (https://www.cnblogs.com/tbiiann/p/6535189.html) 自己实现了最简单版本的lfm (latent factor model)
#   这里我参考这篇文章做自己的lfm 方法，以此为基础加入对bias的处理， 时间变量的处理等

import numpy as np
import math

def LFM_naive(m, k):
    '''
    此函数实现的是最简单的 LFM 功能
    :param m: 表示需要分解的评价矩阵
    :param k: 分解的隐变量个数
    :return:  分解完毕的矩阵 u, v
    '''



if __name__=='__main__':
    pass