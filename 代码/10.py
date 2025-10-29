# -*- coding: utf-8 -*-
"""TESS系外行星检测系统 by 高级统计建模"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ----------------------
# 数据预处理模块
# ----------------------
def load_and_preprocess(file_path):
    # 加载数据
    df = pd.read_excel(file_path, sheet_name='Sheet1')

    # 字段筛选
    columns = [
        'TIC ID', 'TOI', 'RA', 'Dec', 'period', 'duration',
        'Depth (mmag)', 'TESS Mag', 'Stellar Eff Temp (K)',
        'stellar_radius', 'Stellar Mass (M_Sun)', 'disposition',
        'snr', 'radius'  # 使用现成的radius列
    ]
    # 强制检查列存在性
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f"缺失关键列: {missing_cols}")

    df = df[columns].copy()

    df['Depth (ppm)'] = df['Depth (mmag)'] * 1000

    # 数据清洗
    df = df[~df.duplicated(subset=['TIC ID'], keep='first')]  # 去重
    df['disposition'] = df['disposition'].fillna('UNVERIFIED')  # 填充缺失

    # 物理约束过滤
    df = df[
        (df['radius'] <= 30) &  # 使用正确的radius列
        (df['snr'] >= 7)
        ]

    # 异常值处理
    numeric_cols = ['period', 'duration', 'Depth (ppm)', 'TESS Mag']
    df[numeric_cols] = df[numeric_cols].apply(
        lambda x: x.clip(x.quantile(0.05), x.quantile(0.95))
    )

    return df


# ----------------------
# 特征工程模块
# ----------------------
def create_astrophysics_features(df):
    # 行星平衡温度计算
    G = 6.67430e-11  # 引力常数
    df['semi_major_axis'] = ((G * df['Stellar Mass (M_Sun)'] * 1.9885e30 *
                              (df['period'] * 86400) ** 2) / (4 * np.pi ** 2)) ** (1 / 3)
    df['T_eq'] = df['Stellar Eff Temp (K)'] * np.sqrt(
        df['stellar_radius'] * 6.957e8 / (2 * df['semi_major_axis'])
    )

    # 恒星密度计算
    df['stellar_density'] = (3 * df['Stellar Mass (M_Sun)']) / \
                            (4 * np.pi * (df['stellar_radius']) ** 3)

    # 时序统计特征
    df['depth_change_rate'] = df['Depth (ppm)'].rolling(3).std()
    df['phase_smoothness'] = df.groupby('TOI')['Depth (ppm)'].transform(
        lambda x: x.ewm(span=5).mean().diff().std()
    )

    # 交叉特征
    df['snr_depth_ratio'] = df['snr'] / np.sqrt(df['Depth (ppm)'])
    df['mass_radius_ratio'] = df['Stellar Mass (M_Sun)'] / df['stellar_radius'] ** 3

    return df


# ----------------------
# 模型构建模块
# ----------------------
class HybridModel:
    def __init__(self, input_shape):
        self.lgb_model = lgb.LGBMClassifier(
            num_leaves=31, learning_rate=0.05, n_estimators=200
        )
        self.cnn_model = self.build_cnn(input_shape)

    def build_cnn(self, input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Conv1D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.LSTM(16),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def train(self, X_train, y_train, X_val, y_val):
        # LightGBM训练
        self.lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(stopping_rounds=20),
                lgb.log_evaluation(10)
            ]
        )

        # CNN训练
        X_cnn = X_train.values.reshape(-1, X_train.shape[1], 1)
        self.cnn_model.fit(X_cnn, y_train,
                           epochs=50,
                           batch_size=32,
                           validation_split=0.2,
                           callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)])

    def predict(self, X):
        lgb_pred = self.lgb_model.predict_proba(X)[:, 1]
        cnn_pred = self.cnn_model.predict(
            X.values.reshape(-1, X.shape[1], 1)
        ).flatten()
        return 0.6 * lgb_pred + 0.4 * cnn_pred


# ----------------------
# 评估与可视化模块
# ----------------------
def evaluate_model(y_true, y_pred):
    metrics = {
        'AUC': roc_auc_score(y_true, y_pred),
        'F1': f1_score(y_true, (y_pred > 0.5).astype(int)),
        'MCC': matthews_corrcoef(y_true, (y_pred > 0.5).astype(int))
    }
    return metrics


def plot_feature_importance(model, feature_names):
    importance = model.lgb_model.feature_importances_
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance, y=feature_names)
    plt.title('Feature Importance')
    plt.show()


# ----------------------
# 主流程
# ----------------------
if __name__ == "__main__":
    # 数据加载与预处理
    df = load_and_preprocess('tess_raw_data.xlsx')
    # print("Disposition 列的可能取值:", df['disposition'].unique())
    # 特征工程
    df = create_astrophysics_features(df)
    # 确保只保留可处理的disposition类别
    valid_dispositions = ['KP', 'CP', 'PC', 'FP', 'FA', 'APC']
    df = df[df['disposition'].isin(valid_dispositions)]
    # 特征筛选与标准化
    features = ['period', 'duration', 'Depth (ppm)', 'TESS Mag',
                'stellar_density', 'snr_depth_ratio', 'mass_radius_ratio']
    target = df['disposition'].map({'KP': 1, 'CP': 1, 'PC': 1, 'FP': 0, 'FA': 0,'APC': 1})

    scaler = RobustScaler()
    X = scaler.fit_transform(df[features])
    y = target.values.astype(int)
#------------------------------
    # 交叉验证
    skf = StratifiedKFold(n_splits=5)
    auc_scores = []

    # for train_idx, test_idx in skf.split(X, y):
    #     X_train, X_test = X[train_idx], X[test_idx]
    #     y_train, y_test = y[train_idx], y[test_idx]
    #
    #     # 模型训练
    #     model = HybridModel(input_shape=(len(features), 1))
    #     model.train(pd.DataFrame(X_train, columns=features), y_train,
    #                 pd.DataFrame(X_test, columns=features), y_test)
    #
    #     # 预测评估
    #     y_pred = model.predict(pd.DataFrame(X_test, columns=features))
    #     metrics = evaluate_model(y_test, y_pred)
    #     auc_scores.append(metrics['AUC'])
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 检查输入数据
        print("训练数据NaN数量:", np.isnan(X_train).sum())
        print("测试数据NaN数量:", np.isnan(X_test).sum())

        # 模型训练
        model = HybridModel(input_shape=(len(features), 1))
        model.train(pd.DataFrame(X_train, columns=features), y_train,
                    pd.DataFrame(X_test, columns=features), y_test)

        # 分别检查两个模型的预测结果
        lgb_pred = model.lgb_model.predict_proba(pd.DataFrame(X_test, columns=features))[:, 1]
        print("LightGBM预测NaN数量:", np.isnan(lgb_pred).sum())

        cnn_input = pd.DataFrame(X_test, columns=features).values.reshape(-1, len(features), 1)
        cnn_pred = model.cnn_model.predict(cnn_input).flatten()
        print("CNN预测NaN数量:", np.isnan(cnn_pred).sum())

        y_pred = 0.6 * lgb_pred + 0.4 * cnn_pred
        print("混合预测NaN数量:", np.isnan(y_pred).sum())
    # ------------------------------

    print(f"Average AUC: {np.mean(auc_scores):.4f} (±{np.std(auc_scores):.4f})")

    # 可视化
    plot_feature_importance(model, features)