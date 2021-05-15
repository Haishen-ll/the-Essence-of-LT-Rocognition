import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D


def Gaussian_Distribution(N=2, M=1000, m=0, sigma=1):
    '''
    Parameters
    ----------
    N 维度
    M 样本数
    m 样本均值
    sigma: 样本方差

    Returns
    -------
    data  shape(M, N), M 个 N 维服从高斯分布的样本
    Gaussian  高斯分布概率密度函数
    '''
    mean = np.zeros(N) + m  # 均值矩阵，每个维度的均值都为 m
    cov = np.eye(N) * sigma  # 协方差矩阵，每个维度的方差都为 sigma

    # 产生 N 维高斯分布数据
    data = np.random.multivariate_normal(mean, cov, M)
    # N 维数据高斯分布概率密度函数
    Gaussian = multivariate_normal(mean=mean, cov=cov)

    return data, Gaussian

'''一元高斯概率分布图'''
_, Gaussian = Gaussian_Distribution(N=1, M=2000, m=1, sigma=1)
_, Gaussian_2 = Gaussian_Distribution(N=1, M=2000, m=3, sigma=1)
x = np.linspace(-3,5,2000)
x_2 = np.linspace(-1,7,2000)
# 计算一维高斯概率
y = Gaussian.pdf(x)
y_2 = Gaussian.pdf(x_2)
plt.plot(x, y)
plt.plot(x_2, y_2, linestyle='--')
plt.show()
