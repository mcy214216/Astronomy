# -*- coding: utf-8 -*-
"""
基于物理约束深度学习的系外行星宜居性预测系统
环境要求：Python 3.8+ | 需要安装的库：
!pip install pandas numpy matplotlib seaborn tensorflow scikit-learn imbalanced-learn shap openpyxl
"""

import os
import io
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
import shap
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# ======================
# 数据获取模块
# ======================
# def fetch_data():
#     """获取系外行星数据，优先从API获取，失败时使用本地备份"""
#     # try:
#     #     api_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
#     #     params = {
#     #         "query": "select pl_name,pl_rade,pl_orbper,st_teff,st_rad,sy_dist,pl_eqt from pscomppars",
#     #         "format": "csv"
#     #     }
#     #     response = requests.get(api_url, params=params, timeout=30)
#     #     df = pd.read_csv(io.StringIO(response.text))
#     #     df.to_excel("raw_data.xlsx", index=False)  # 保存原始数据
#     #     print("API数据获取成功，已保存到raw_data.xlsx")
#     # except Exception as e:
#     #     print(f"API获取失败: {str(e)}，使用本地备份数据")
#     #     df = pd.read_excel(
#     #         "https://docs.google.com/spreadsheets/d/1cvZ4JhZT1uKhK9Bk0pSXo3Mx4yhDq7J7/export?format=xlsx")
#     df = pd.read_excel("raw_data.xlsx")
#
#     return df


# ======================
# 数据预处理模块
# ======================
def preprocess_data(df):
    """数据清洗与特征工程"""
    # 缺失值处理
    df = df.dropna(subset=['pl_rade', 'st_teff', 'st_rad', 'pl_orbper'])

    # 异常值过滤
    df = df[(df['pl_rade'] > 0.3) & (df['pl_rade'] < 20)]
    df = df[df['st_teff'] > 2500]

    # 计算恒星光度（斯特藩-玻尔兹曼定律）
    df['L_star'] = 4 * np.pi * (df['st_rad'] ** 2) * 5.67e-8 * (df['st_teff'] ** 4)

    # 生成宜居标签（基于宜居带计算）
    inner = 0.75 * np.sqrt(df['L_star'])
    outer = 1.77 * np.sqrt(df['L_star'])
    df['habitable'] = ((df['pl_orbper'] > inner) & (df['pl_orbper'] < outer)).astype(int)

    # 特征选择
    features = ['pl_rade', 'pl_orbper', 'st_teff', 'st_rad', 'sy_dist']
    target = 'habitable'

    return df[features + [target]], features


# ======================
# 物理约束层实现
# ======================
class PhysicsConstraint(Layer):
    """自定义物理约束层"""

    def __init__(self, **kwargs):
        super(PhysicsConstraint, self).__init__(**kwargs)

    def call(self, inputs):
        T_pred, R_star = inputs
        L_pred = 4 * np.pi * (R_star ** 2) * 5.67e-8 * (T_pred ** 4)
        return L_pred


# ======================
# 模型构建模块
# ======================
def build_model(input_dim):
    """构建混合物理约束模型"""
    inputs = Input(shape=(input_dim,))

    # 共享特征层
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)

    # 温度预测分支
    T_pred = Dense(1, activation='linear', name='temp_output')(x)

    # 物理约束分支
    physics_output = PhysicsConstraint(name='physics_output')([T_pred, inputs[:, 3]])

    model = Model(inputs=inputs, outputs=[T_pred, physics_output])
    return model


# ======================
# 主程序
# ======================
if __name__ == "__main__":
    # 数据准备阶段
    os.makedirs("output", exist_ok=True)
    raw_df = pd.read_excel("raw_data.xlsx")
    processed_df, features = preprocess_data(raw_df)

    # 保存预处理数据
    processed_df.to_excel("processed_data.xlsx", index=False)

    # 数据分割
    X = processed_df[features].values
    y = processed_df['habitable'].values

    # 处理类别不平衡
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )

    # 模型构建
    model = build_model(X_train.shape[1])


    # 自定义混合损失函数
    def hybrid_loss(y_true, y_pred):
        mse_loss = tf.keras.losses.mean_squared_error(y_true[0], y_pred[0])
        physics_loss = tf.reduce_mean(tf.square(y_pred[1] - y_true[1]))
        return mse_loss + 0.5 * physics_loss


    # 模型编译
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={'temp_output': 'mse', 'physics_output': hybrid_loss},
        metrics={'temp_output': 'mae'}
    )

    # 模型训练
    history = model.fit(
        X_train, {'temp_output': y_train, 'physics_output': y_train},
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
    )

    # 模型评估
    y_pred = model.predict(X_test)[0].flatten()
    y_pred_class = (y_pred > 0.5).astype(int)

    # 保存预测结果
    result_df = pd.DataFrame({
        '实际值': y_test,
        '预测概率': y_pred,
        '预测类别': y_pred_class
    })
    result_df.to_excel("output/predictions.xlsx", index=False)

    # 评估指标
    report = classification_report(y_test, y_pred_class, output_dict=True)
    pd.DataFrame(report).transpose().to_excel("output/evaluation_metrics.xlsx")

    # 可解释性分析
    explainer = shap.DeepExplainer(model, X_train[:100])
    shap_values = explainer.shap_values(X_test[:50])

    plt.figure()
    shap.summary_plot(shap_values[0], X_test[:50], feature_names=features, show=False)
    plt.savefig("output/shap_summary.png", dpi=300, bbox_inches='tight')

    # 训练过程可视化
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型训练曲线')
    plt.legend()

    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_pred):.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC曲线')
    plt.legend()

    plt.savefig("output/training_metrics.png", dpi=300)

    print("""
    ======================
    执行完成！生成文件清单：
    - raw_data.xlsx        原始数据
    - processed_data.xlsx  预处理数据
    - output/predictions.xlsx     预测结果
    - output/evaluation_metrics.xlsx 评估指标
    - output/shap_summary.png     特征重要性
    - output/training_metrics.png 训练过程
    ======================
    """)