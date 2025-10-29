import numpy as np
import pandas as pd
import emcee
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ========== 数据加载 ==========
# 读取 Excel 文件
excel_path = "Pantheon+SH0ES.xlsx"  # 替换为你的 Excel 文件路径
df = pd.read_excel(excel_path)

# 提取关键列
z_data = df["zHD"].astype(float)  # 红移
mu_data = df["MU_SH0ES"].astype(float)  # 距离模数
mu_err_stat = df["MU_SH0ES_ERR_DIAG"].astype(float)  # 统计误差
mu_err_sys = df["m_b_corr_err_DIAG"].astype(float)  # 系统误差

# 合并总误差
mu_err = np.sqrt(mu_err_stat ** 2 + mu_err_sys ** 2)

# 过滤无效数据
mask = (z_data > 0.001) & (~z_data.isna()) & (~mu_data.isna())
z_data = z_data[mask].values
mu_data = mu_data[mask].values
mu_err = mu_err[mask].values

print(f"成功加载 {len(z_data)} 个有效数据点")


# ========== 宇宙学模型 ==========
def mu_theory(z, H0, Om):
    c = 299792.458  # 光速 (km/s)
    Omega_lambda = 1 - Om  # 暗能量密度参数

    # 计算共动距离 D_C
    def E(z_prime):
        return np.sqrt(Om * (1 + z_prime) ** 3 + Omega_lambda)

    D_C = np.array([quad(lambda x: 1 / E(x), 0, zi)[0] for zi in z])
    D_L = (1 + z) * D_C * c / H0  # 光度距离
    return 5 * np.log10(D_L) + 25  # 距离模数


# ========== 统计推断 ==========
# 对数似然函数
def log_likelihood(theta):
    H0, Om = theta
    mu_pred = mu_theory(z_data, H0, Om)
    return -0.5 * np.sum(((mu_data - mu_pred) / mu_err) ** 2)


# 先验分布
def log_prior(theta):
    H0, Om = theta
    if 60 < H0 < 80 and 0.2 < Om < 0.5:  # 合理范围
        return 0.0
    return -np.inf


# 后验概率
def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


# ========== MCMC 采样 ==========
# 设置 MCMC 参数
nwalkers = 32  # 采样器数量
ndim = 2  # 参数维度 (H0, Om)
nsteps = 3000  # 采样步数

# 初始猜测
initial = np.array([73.0, 0.3])  # 初始值 (H0, Om)
initial_guesses = initial + 1e-3 * np.random.randn(nwalkers, ndim)

# 运行 MCMC 采样器
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
sampler.run_mcmc(initial_guesses, nsteps, progress=True)

# 获取采样链
samples = sampler.get_chain(discard=1000, thin=20, flat=True)

# ========== 结果分析 ==========
# 参数估计
H0_mean = np.mean(samples[:, 0])
Om_mean = np.mean(samples[:, 1])
print(f"H0 = {H0_mean:.1f} ± {np.std(samples[:, 0]):.1f} km/s/Mpc")
print(f"Ωm = {Om_mean:.3f} ± {np.std(samples[:, 1]):.3f}")

# 后验分布图
import corner

fig = corner.corner(samples, labels=["H0", "Ωm"], truths=[H0_mean, Om_mean])
plt.show()