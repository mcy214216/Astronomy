# 《基于TESS数据的统计增强行星检测模型》
# -*- coding: utf-8 -*-
# 环境依赖：pip install pandas openpyxl scikit-learn tensorflow matplotlib

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, f1_score,
                             accuracy_score, confusion_matrix,
                             classification_report)
from sklearn.ensemble import GradientBoostingClassifier
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# TESS数据配置
# ======================
TESS_COLUMN_MAP = {
    # 基础特征
    'Period (days)': 'period',
    'Duration (hours)': 'duration',
    'Depth (ppm)': 'depth',
    'Planet Radius (R_Earth)': 'radius',
    'Equilibrium Temperature (K)': 'temp',
    'Stellar Radius (R_Sun)': 'stellar_radius',
    'Planet SNR': 'snr',

    # 标签相关
    'TFOPWG Disposition': 'disposition',
    'Comments': 'comments'
}

SELECTED_FEATURES = [
    'period', 'duration', 'depth', 'radius',
    'temp', 'stellar_radius', 'snr'
]


# ======================
# 数据获取与存储模块（修改版）
# ======================
def fetch_tess_data():
    """从本地Excel文件读取TESS候选行星数据"""
    try:
        # 读取本地数据
        df = pd.read_excel("tess_raw_data.xlsx", engine='openpyxl')
        print("原始数据维度:", df.shape)

        # 列名转换
        df = df.rename(columns=TESS_COLUMN_MAP)
        return df

    except Exception as e:
        print(f"数据读取失败: {str(e)}（请确认tess_raw_data.xlsx在当前目录）")
        return None


# ======================
# 数据预处理模块（保持不变）
# ======================
def preprocess_tess_data(df):
    """TESS数据专用预处理"""
    try:
        # 数据清洗
        df = df[df['disposition'].isin(['Confirmed', 'False Positive'])].copy()
        df = df.dropna(subset=SELECTED_FEATURES)

        # 标签编码
        df['label'] = df['disposition'].map({'Confirmed': 1, 'False Positive': 0})

        # 特征工程
        X = df[SELECTED_FEATURES].copy()

        # 构造统计特征
        X['period_depth_ratio'] = X['period'] / (X['depth'] + 1e-6)
        X['temp_radius_product'] = X['temp'] * X['radius']
        X['duration_norm'] = X['duration'] / X['period']

        # 处理异常值
        X = X[(X['period'] > 0.5) & (X['period'] < 500)]
        X = X[(X['depth'] > 10) & (X['depth'] < 1e5)]

        # 标准化处理
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        return X_scaled, df['label'], df

    except Exception as e:
        print(f"数据处理失败: {str(e)}")
        return None, None, None


# ======================
# 混合模型构建模块
# ======================
def build_tess_model(input_shape):
    """构建TESS专用混合模型"""
    # GBDT模型
    gb_model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        min_samples_leaf=10
    )

    # 深度学习模型
    dl_model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # 模型融合
    inputs = tf.keras.Input(shape=(input_shape,))
    gb_out = tf.keras.layers.Lambda(lambda x: gb_model.predict_proba(x)[:, 1])(inputs)
    dl_out = dl_model(inputs)
    combined = tf.keras.layers.Concatenate()([gb_out, dl_out])
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(combined)

    ensemble_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    ensemble_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )
    return ensemble_model, gb_model, dl_model


# ======================
# 增强可视化模块
# ======================
def generate_tess_visualizations(history, metrics, model, features):
    """生成TESS专用分析图表"""
    plt.figure(figsize=(15, 10))

    # 学习曲线
    plt.subplot(2, 2, 1)
    plt.plot(history.history['auc'], label='Train')
    plt.plot(history.history['val_auc'], label='Validation')
    plt.title('Model AUC Curve')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()

    # 特征重要性
    plt.subplot(2, 2, 2)
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:]
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.title('Feature Importances')

    # 混淆矩阵
    plt.subplot(2, 2, 3)
    sns.heatmap(metrics['Confusion_Matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # 保存综合图表
    plt.tight_layout()
    plt.savefig('tess_analysis.png')
    plt.close()

# ======================
# 主程序流程
# ======================
if __name__ == "__main__":
    # 数据获取
    raw_df = fetch_tess_data()
    if raw_df is None:
        exit()

    # 数据预处理
    X, y, processed_df = preprocess_tess_data(raw_df)
    if X is None:
        exit()

    print(f"\n可用样本量: {len(X)}")
    print("特征列表:", X.columns.tolist())
    print("标签分布:\n", y.value_counts())

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        stratify=y,
        random_state=42
    )

    # 模型训练
    ensemble_model, gb_model, _ = build_tess_model(X_train.shape[1])

    print("\n训练GBDT模型...")
    gb_model.fit(X_train, y_train)

    print("\n训练混合模型...")
    history = ensemble_model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=80,
        batch_size=64,
        class_weight={0: 0.7, 1: 1.3},  # 处理类别不平衡
        verbose=1
    )

    # 模型评估
    y_pred = ensemble_model.predict(X_test).flatten()
    metrics = {
        'AUC': roc_auc_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, (y_pred > 0.5).astype(int)),
        'Accuracy': accuracy_score(y_test, (y_pred > 0.5).astype(int)),
        'Confusion_Matrix': confusion_matrix(y_test, (y_pred > 0.5).astype(int)),
        'Classification_Report': classification_report(y_test, (y_pred > 0.5).astype(int))
    }

    # 结果保存
    result_df = pd.DataFrame({
        'Predicted': (y_pred > 0.5).astype(int),
        'Probability': y_pred,
        'Actual': y_test.values
    }, index=X_test.index)

    with pd.ExcelWriter('tess_results.xlsx') as writer:
        result_df.to_excel(writer, sheet_name='Predictions')
        pd.DataFrame(metrics['Classification_Report']).to_excel(
            writer, sheet_name='Metrics'
        )

    # 可视化
    generate_tess_visualizations(history, metrics, gb_model, X.columns)

    print("\n=== 最终模型表现 ===")
    print(f"AUC: {metrics['AUC']:.4f}")
    print(f"F1-Score: {metrics['F1-Score']:.4f}")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print("\n分类报告:")
    print(metrics['Classification_Report'])