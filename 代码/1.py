# 创建时间   : 2025/3/23 02:07
# 作者      : 叶之瞳
# 文件名     : 1.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import emcee
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from scipy.interpolate import CubicSpline
import corner
#--------------------
# 读取Planck CMB温度功率谱
# L：多极矩
# TT：温度涨落的自相关谱
# TE：温度与E模式极化的交叉谱
# EE：E模式极化的自相关谱
# BB：B模式极化的自相关谱
# PP：引力透镜势的功率谱
planck_data = pd.read_csv('COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt',
                        delim_whitespace=True, skiprows=3)
ell = planck_data['L'].values# 多极矩
D_ell = planck_data['PP'].values  # C_ell * ell(ell+1)/(2π)# 功率谱值
# sigma = planck_data['sigma'].values# 误差

# 去噪（小波变换）
import pywt
coeffs = pywt.wavedec(D_ell, 'db4', level=5)# 小波变换
sigma_noise = np.median(np.abs(coeffs[-1])) / 0.6745#
coeffs_thresh = [pywt.threshold(c, sigma_noise*2, mode='soft') for c in coeffs]
D_ell_clean = pywt.waverec(coeffs_thresh, 'db4')
# 绘制图像
plt.figure(figsize=(10, 6))
plt.plot(ell, D_ell, label='Original')
plt.plot(ell, D_ell_clean, label='Cleaned')
plt.xlabel('Multipole')
plt.ylabel('C_ell')
plt.legend()
plt.show()
