# 基于统计特征增强的轻量化行星检测模型
### Lightweight Exoplanet Detection Model with Statistical Feature Enhancement

一个基于 **TESS 天文时序数据** 的系外行星候选体检测项目。  
本项目融合 **DNN（深度神经网络）** 与 **LightGBM**，结合天体物理先验和统计特征增强方法，实现对系外行星信号的高效识别与分类。

---

## 项目简介

在系外行星检测任务中，传统方法通常依赖阈值判定、人工特征提取或经典统计检验，这类方法在面对高噪声、弱信号、类别不平衡等问题时，往往存在误检率高、对小信号不敏感、计算效率不足等局限。

本项目基于 **NASA TESS（凌星系外行星巡天卫星）** 发布的候选行星数据集，提出一种 **统计特征增强 + 轻量化混合模型** 的检测方案。核心思想是：

- 通过 **天体物理特征工程** 提升弱信号表达能力
- 使用 **DNN** 提取非线性深层特征
- 使用 **LightGBM** 完成高效分类决策
- 利用 **SHAP** 分析特征重要性，提升模型可解释性

实验结果表明，该模型在 TESS 数据集上取得了较好的检测性能，平均 **AUC = 0.822 ± 0.060**、**F1-score = 0.918 ± 0.015**、**Precision = 0.906 ± 0.019**、**Accuracy = 0.858 ± 0.026**。

---

## 项目特点

- **轻量化混合架构**：融合 DNN 与 LightGBM，兼顾特征表达能力与训练效率。
- **统计特征增强**：构建轨道参数、恒星结构和时序动态特征，增强对微弱行星信号的识别能力。
- **鲁棒数据预处理**：采用缺失值清洗、物理约束过滤、改进 IQR 截断与 RobustScaler 标准化。
- **可解释性分析**：引入 SHAP 对特征贡献度进行分析，辅助理解模型决策机制。
- **适合扩展部署**：可用于天文巡天任务中大规模光变曲线的自动筛选。

---

## 数据来源

本项目使用 **NASA TESS 候选行星数据集**，通过科学数据接口 API 获取原始观测记录，共约 **6397 条样本**。数据包含：

- 光变曲线时序特征
- 恒星参数（如恒星半径、有效温度）
- 行星候选体属性（如轨道周期、行星半径比）

数据集中类别不平衡明显，疑似真实行星信号占比较低，因此在建模过程中加入了类别不平衡处理与特征增强策略。

---

## 数据预处理流程

项目中的数据预处理主要包括以下步骤：

### 1. 字段筛选
从原始表中保留核心字段，例如：

- `period`
- `duration`
- `Depth (mmag)`
- `TESS Mag`
- `Stellar Eff Temp (K)`
- `stellar_radius`
- `Stellar Mass (M_Sun)`
- `disposition`
- `snr`
- `radius`

对应代码中通过 `load_and_preprocess()` 实现。

### 2. 缺失值处理
- 删除关键字段缺失样本
- 将 `disposition` 空值填充为 `UNVERIFIED`

### 3. 物理约束过滤
依据天体物理先验去除非法样本，例如：

- `radius` ∈ (0.1, 30)
- `snr >= 7`
- `period > 0`
- `stellar_radius > 0`
- `Stellar Mass (M_Sun) > 0`

### 4. 异常值处理
对 `period`、`duration`、`TESS Mag` 等连续变量采用改进 IQR 方法进行 **5%–95% 分位数截断**。

### 5. 标准化
采用 `RobustScaler` 对特征进行鲁棒标准化，以减小异常值对模型训练的影响。

---

## 特征工程

为了提升模型对弱信号和复杂噪声环境的适应能力，项目构建了以下天体物理增强特征：

### 1. 轨道参数重构
- **semi_major_axis**：依据开普勒第三定律计算轨道半长轴
- **T_eq**：计算行星平衡温度

### 2. 恒星结构特征
- **stellar_density**：根据恒星质量和半径计算平均密度

### 3. 时序动态特征
- **depth_change_rate**：利用滚动窗口计算凌星深度变化率
- **snr_depth_ratio**：信噪比与光变深度的比值，用于衡量信号显著性

最终模型保留的核心特征包括：

- `period`
- `T_eq`
- `stellar_density`
- `snr_depth_ratio`
- `TESS Mag`
- `duration`
- `depth_change_rate`

---

## 模型架构

本项目采用 **DNN + LightGBM** 的双阶段混合建模框架。

### DNN 模块
DNN 负责对输入特征进行非线性抽象，主要结构如下：

- 输入层：7维核心特征
- 隐藏层1：64 神经元 + ReLU + BatchNorm + Dropout(0.3)
- 隐藏层2：32 神经元 + ReLU
- 输出层：1 神经元 + Sigmoid

优化器使用 **Adam**，学习率为 `1e-3`。

### LightGBM 模块
LightGBM 用于结构化特征分类，主要参数包括：

- `num_leaves = 31`
- `max_depth = 6`
- `learning_rate = 0.05`
- `n_estimators = 200`
- `min_child_samples = 20`
- `reg_alpha = 0.1`
- `reg_lambda = 0.1`
- `subsample = 0.8`
- `colsample_bytree = 0.7`

参数优化采用贝叶斯优化思想确定。

### 融合策略
最终预测结果采用加权融合方式：

```python
final_pred = 0.6 * lgb_pred + 0.4 * dnn_pred