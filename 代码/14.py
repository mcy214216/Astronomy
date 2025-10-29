# 创建时间   : 2025/4/11 01:43
# 作者      : 叶之瞳
# 文件名     : 14.py
# 创建时间   : 2025/3/26 02:59
# 作者      : 叶之瞳
# 文件名     : 11.py
# -*- coding: utf-8 -*-
"""改进版TESS系外行星检测系统"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             accuracy_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# ----------------------
# 数据预处理模块（增强版）
# ----------------------
def load_and_preprocess(file_path):
    # 加载数据
    df = pd.read_excel(file_path, sheet_name='Sheet1')

    # 字段筛选与类型转换
    columns = [
        'TIC ID', 'TOI', 'RA', 'Dec', 'period', 'duration',
        'Depth (mmag)', 'TESS Mag', 'Stellar Eff Temp (K)',
        'stellar_radius', 'Stellar Mass (M_Sun)', 'disposition',
        'snr', 'radius'
    ]
    df = df[columns].copy()

    # 数据清洗增强
    df = df.dropna(subset=['disposition', 'period', 'TESS Mag'])  # 关键字段去NaN
    df['disposition'] = df['disposition'].fillna('UNVERIFIED')

    # 物理约束过滤（增加边界检查）
    df = df[
        (df['radius'].between(0.1, 30)) &
        (df['snr'] >= 7) &
        (df['period'] > 0) &
        (df['Stellar Mass (M_Sun)'] > 0) &
        (df['stellar_radius'] > 0)
        ]

    # 异常值处理（使用IQR方法）
    numeric_cols = ['period', 'duration', 'TESS Mag']
    for col in numeric_cols:
        q1 = df[col].quantile(0.05)
        q3 = df[col].quantile(0.95)
        df[col] = df[col].clip(q1, q3)

    return df


# ----------------------
# 特征工程模块（安全计算）
# ----------------------
def create_astrophysics_features(df):
    # 行星平衡温度计算（防止无效计算）
    G = 6.67430e-11
    df['semi_major_axis'] = (
                                    (G * df['Stellar Mass (M_Sun)'].clip(lower=1e-6) * 1.9885e30 *
                                     (df['period'].clip(lower=1e-6) * 86400) ** 2) /
                                    (4 * np.pi ** 2)
                            ) ** (1 / 3)

    df['T_eq'] = df['Stellar Eff Temp (K)'] * np.sqrt(
        df['stellar_radius'].clip(lower=1e-6) * 6.957e8 /
        (2 * df['semi_major_axis'].replace(0, np.nan))
    )

    # 恒星密度计算（避免除以零）
    df['stellar_density'] = (
            (3 * df['Stellar Mass (M_Sun)']) /
            (4 * np.pi * df['stellar_radius'].clip(lower=1e-6) ** 3)
    )

    # 时序特征（填充NaN）
    df['depth_change_rate'] = (
        df['Depth (mmag)'].rolling(3, min_periods=1).std().fillna(0)
    )

    # 交叉特征（标准化处理）
    df['snr_depth_ratio'] = (
            df['snr'] / np.sqrt(df['Depth (mmag)'].clip(lower=1e-6))
    )

    # 清理可能的NaN
    df = df.dropna(subset=['stellar_density', 'T_eq'])

    return df


# ----------------------
# 模型构建模块（使用DNN替代CNN）
# ----------------------
class HybridModel:
    def __init__(self, input_shape):
        self.lgb_model = lgb.LGBMClassifier(
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=200,
            verbosity=-1 , # 关闭LightGBM警告
            reg_alpha = 0.1,  # 添加L1正则化
            reg_lambda = 0.1,  # 添加L2正则化
            class_weight = 'balanced'  # 处理类别不平衡
        )
        self.dnn_model = self.build_dnn(input_shape)

    def build_dnn(self, input_shape):
        """更稳健的DNN结构"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, X_train, y_train, X_val, y_val):
        # LightGBM训练
        self.lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(stopping_rounds=20),
                lgb.log_evaluation(0)  # 关闭训练日志
            ]
        )

        # DNN训练（添加梯度裁剪）
        self.dnn_model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ],
            verbose=0
        )

    def predict(self, X):
        lgb_pred = self.lgb_model.predict_proba(X)[:, 1]
        dnn_pred = self.dnn_model.predict(X, verbose=0).flatten()
        return 0.6 * lgb_pred + 0.4 * dnn_pred


# ----------------------
# 数据可视化模块（新增部分）
# ----------------------
def data_visualization(df):
    plt.figure(figsize=(15, 12))

    # 1. 目标变量分布
    plt.subplot(3, 2, 1)
    target_dist = df['disposition'].value_counts()
    sns.barplot(x=target_dist.index, y=target_dist.values, hue=target_dist.index,
                palette="viridis", legend=False)
    plt.title('目标变量分布（disposition）')
    plt.xticks(rotation=45)

    # 2. 关键数值特征分布
    plt.subplot(3, 2, 2)
    sns.histplot(df['period'], kde=True, bins=30, color='royalblue')
    plt.title('轨道周期分布（period）')
    plt.xlabel('Days')

    plt.subplot(3, 2, 3)
    sns.histplot(df['TESS Mag'], kde=True, bins=30, color='darkorange')
    plt.title('TESS星等分布')
    plt.xlabel('Magnitude')

    plt.subplot(3, 2, 4)
    sns.boxplot(x='disposition', y='radius', data=df, hue='disposition',
                palette="Set2", legend=False)
    plt.title('不同类别行星半径分布')
    plt.xticks(rotation=45)

    # 3. 特征相关性热力图
    plt.subplot(3, 2, 5)
    numeric_cols = ['period', 'duration', 'Depth (mmag)', 'TESS Mag',
                    'Stellar Eff Temp (K)', 'snr', 'radius']
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('特征相关性矩阵')

    # 4. 物理参数散点图
    plt.subplot(3, 2, 6)
    sns.scatterplot(x='Stellar Eff Temp (K)', y='radius',
                    hue='disposition', data=df,
                    palette='Set1', alpha=0.7)
    plt.title('恒星温度 vs 行星半径')
    plt.xscale('log')

    plt.tight_layout()
    plt.savefig('data_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

# ----------------------
# 主流程（增强稳定性）
# ----------------------
if __name__ == "__main__":
    # 数据加载与预处理
    df = load_and_preprocess('tess_raw_data.xlsx')
    print(f"初始数据量: {len(df)}")
    data_visualization(df)

    # 特征工程
    df = create_astrophysics_features(df)
    print(f"特征工程后数据量: {len(df)}")

    # 目标编码
    target_map = {'KP': 1, 'CP': 1, 'PC': 1, 'APC': 1, 'FP': 0, 'FA': 0}
    df = df.assign(target=df['disposition'].map(target_map))

    # 特征选择与标准化
    features = ['period', 'duration', 'Depth (mmag)', 'TESS Mag',
                'stellar_density', 'snr_depth_ratio', 'T_eq']
    X = df[features]
    y = df['target']
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # 交叉验证流程增强
    skf = StratifiedKFold(n_splits=50, shuffle=True, random_state=42)
    auc_scores, f1_scores = [], []
    precision_scores, accuracy_scores = [], []
    all_y_test, all_y_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
        print(f"\n=== Fold {fold + 1} ===")

        # 数据分割
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 模型初始化与训练
        model = HybridModel(input_shape=(X_train.shape[1],))
        model.train(X_train, y_train, X_test, y_test)

        # 预测与评估
        y_pred = model.predict(X_test)
        y_pred_class = (y_pred >= 0.5).astype(int)

        # 存储全局预测结果
        all_y_test.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        # 计算各项指标
        fold_auc = roc_auc_score(y_test, y_pred)
        fold_f1 = f1_score(y_test, y_pred_class)
        fold_precision = precision_score(y_test, y_pred_class)
        fold_accuracy = accuracy_score(y_test, y_pred_class)

        # 记录指标
        auc_scores.append(fold_auc)
        f1_scores.append(fold_f1)
        precision_scores.append(fold_precision)
        accuracy_scores.append(fold_accuracy)

        print(f"""Fold {fold + 1} 评估结果:
        AUC: {fold_auc:.4f}
        F1-score: {fold_f1:.4f}
        精确率: {fold_precision:.4f}
        准确率: {fold_accuracy:.4f}""")

    # 最终评估结果
    print("\n=== 最终评估结果 ===")
    print(f"平均 AUC: {np.mean(auc_scores):.4f} (±{np.std(auc_scores):.4f})")
    print(f"平均 F1-score: {np.mean(f1_scores):.4f} (±{np.std(f1_scores):.4f})")
    print(f"平均 精确率: {np.mean(precision_scores):.4f} (±{np.std(precision_scores):.4f})")
    print(f"平均 准确率: {np.mean(accuracy_scores):.4f} (±{np.std(accuracy_scores):.4f})")

    # ----------------------
    # 增强可视化部分
    # ----------------------
    # 1. 多指标趋势图
    plt.figure(figsize=(14, 6))
    metrics = [auc_scores, f1_scores, precision_scores, accuracy_scores]
    metric_names = ['AUC', 'F1-score', 'Precision', 'Accuracy']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, (values, name, color) in enumerate(zip(metrics, metric_names, colors)):
        plt.plot(range(1, 51), values, 'o--', color=color,
                 label=f'{name} (平均: {np.mean(values):.2f}±{np.std(values):.2f})')

    plt.title('各Fold指标表现趋势', fontsize=14)
    plt.xlabel('Fold编号', fontsize=12)
    plt.ylabel('得分', fontsize=12)
    plt.xticks(range(1, 51))
    plt.ylim(0.5, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('metrics_trend.png', dpi=300, bbox_inches='tight')

    # 2. 平均指标柱状图
    plt.figure(figsize=(10, 6))
    means = [np.mean(metric) for metric in metrics]
    stds = [np.std(metric) for metric in metrics]
    x_pos = np.arange(len(metric_names))

    plt.bar(x_pos, means, yerr=stds, align='center', alpha=0.7,
            capsize=10, color=colors)
    plt.xticks(x_pos, metric_names)
    plt.ylabel('平均得分')
    plt.title('交叉验证平均指标表现')
    plt.ylim(0.0, 1.0)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('average_metrics.png', dpi=300)

    # 3. 混淆矩阵
    plt.figure(figsize=(8, 6))
    y_pred_class = (np.array(all_y_pred) >= 0.5).astype(int)
    cm = confusion_matrix(all_y_test, y_pred_class)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['非行星', '候选行星'],
                yticklabels=['真实非行星', '真实候选行星'])
    plt.title('全局混淆矩阵', fontsize=14)
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)

    plt.show()