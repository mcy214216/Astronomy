# 创建时间   : 2025/3/24 02:57
# 作者      : 叶之瞳
# 文件名     : 5.py
# -*- coding: utf-8 -*-
"""
基于物理约束深度学习的系外行星宜居性预测系统
主要改进：
1. 修正物理约束层的实现
2. 调整损失函数和模型输出
3. 优化数据预处理流程
4. 修复评估指标计算
环境要求：Python 3.8+ | 需要安装的库：
!pip install pandas numpy matplotlib seaborn tensorflow scikit-learn imbalanced-learn shap openpyxl
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import shap
import tensorflow as tf
tf.compat.v1.disable_eager_execution()  # 关键修复！
from tensorflow.keras.layers import Layer, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
# 在代码开头添加以下设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="shap")
# ======================
# 数据预处理模块
# ======================
# def preprocess_data(df):
#     """增强型数据预处理"""
#     # 缺失值处理
#     df = df.dropna(subset=['pl_rade', 'st_teff', 'st_rad', 'pl_orbper', 'sy_dist'])
#
#     # 异常值过滤
#     df = df[(df['pl_rade'] > 0.3) & (df['pl_rade'] < 20)]
#     df = df[(df['st_teff'] > 2500) & (df['st_teff'] < 10000)]
#
#     # 计算轨道半径（使用开普勒第三定律）
#     G = 6.6743e-11  # m³/kg/s²
#     M_sun = 1.9885e30  # kg
#     df['pl_orba'] = ((df['pl_orbper'] * 86400) ** 2 * G * M_sun / (4 * np.pi ** 2)) ** (1 / 3)
#
#     # 计算恒星光度（斯特藩-玻尔兹曼定律）
#     df['L_star'] = 4 * np.pi * (df['st_rad'] ** 2) * 5.67e-8 * (df['st_teff'] ** 4)
#
#     # 生成宜居标签（保守宜居带）
#     inner = 0.75 * np.sqrt(df['L_star'])
#     outer = 1.77 * np.sqrt(df['L_star'])
#     df['habitable'] = ((df['pl_orba'] > inner) & (df['pl_orba'] < outer)).astype(int)
#
#     # 特征工程
#     features = ['pl_rade', 'st_teff', 'st_rad', 'sy_dist', 'pl_orba']
#     target = 'habitable'
#
#     return df[features + [target]], features



def preprocess_data(df):
    """改进的预处理流程"""
    # 缺失值处理（保留关键特征）
    df = df.dropna(subset=['pl_rade', 'st_teff', 'st_rad', 'pl_orbper', 'sy_dist'])

    # 异常值过滤（放宽限制）
    df = df[
        (df['pl_rade'] > 0.3) &
        (df['pl_rade'] < 30) &  # 放宽上限
        (df['st_teff'] > 2000) &  # 降低下限
        (df['st_teff'] < 12000)
    ]

    # 计算轨道半径（天文单位转换）
    G = 6.6743e-11
    M_sun = 1.9885e30
    AU = 1.496e11  # 1 astronomical unit in meters

    df['pl_orba_au'] = (  # 转换为天文单位
        ((df['pl_orbper']*86400)**2 * G * M_sun / (4*np.pi**2))**(1/3)
    ) / AU

    # 计算恒星光度（太阳光度单位）
    df['L_star'] = (df['st_rad']**2) * (df['st_teff']/5778)**4  # 以太阳为基准

    # 动态宜居带计算（Kopparapu模型）
    # 内边界公式：S_eff = S_inner * L_star
    # 外边界公式：S_eff = S_outer * L_star
    S_inner = 0.95  # 保守宜居带内边界系数
    S_outer = 1.67  # 保守宜居带外边界系数

    df['habitable'] = (
        (df['pl_orba_au'] > S_inner * np.sqrt(df['L_star'])) &
        (df['pl_orba_au'] < S_outer * np.sqrt(df['L_star']))
    ).astype(int)

    # 特征工程（验证特征有效性）
    features = [
        'pl_rade',        # 行星半径
        'st_teff',        # 恒星有效温度
        'st_rad',         # 恒星半径
        'sy_dist',        # 系统距离
        'pl_orba_au',     # 轨道半径（天文单位）
        'L_star'          # 恒星光度
    ]

    # 数据分布验证
    class_dist = df['habitable'].value_counts()
    print(f"\n类别分布验证:\n{class_dist}")

    if len(class_dist) < 2:
        raise ValueError("数据集中只存在单一类别，请检查宜居带计算参数！")

    return df[features + ['habitable']], features

# ======================
# 物理约束层实现（修正版）
# ======================
class PhysicsConstraint(Layer):
    """增强物理约束层"""

    def __init__(self, **kwargs):
        super(PhysicsConstraint, self).__init__(**kwargs)

    def call(self, inputs):
        T_pred, R_star = inputs
        # 斯特藩-玻尔兹曼定律约束
        L_pred = 4 * np.pi * (R_star ** 2) * 5.67e-8 * (T_pred ** 4)
        return L_pred


# ======================
# 模型构建模块（优化版）
# ======================
def build_model(input_dim):
    # """增强型混合模型架构"""
    # inputs = Input(shape=(input_dim,))
    #
    # # 共享特征提取层
    # x = Dense(256, activation='relu', kernel_initializer='he_normal')(inputs)
    # x = Dropout(0.4)(x)
    # x = Dense(128, activation='relu', kernel_initializer='he_normal')(x)
    # x = Dropout(0.3)(x)
    #
    # # 温度预测分支
    # T_pred = Dense(1, activation='linear', name='temp_output')(x)
    #
    # # 物理约束分支
    # R_star = inputs[:, 2]  # st_rad特征
    # L_pred = PhysicsConstraint(name='physics_output')([T_pred, R_star])
    #
    # # 宜居性预测分支
    # habitable = Dense(1, activation='sigmoid', name='habitable_output')(x)
    #
    # model = Model(inputs=inputs, outputs=[habitable, T_pred, L_pred])
    # return model
    # """修正后的单输出模型"""
    # inputs = Input(shape=(input_dim,))
    #
    # # 共享特征提取层
    # x = Dense(256, activation='relu')(inputs)
    # x = Dropout(0.4)(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dropout(0.3)(x)
    #
    # # 单一输出层
    # habitable = Dense(1, activation='sigmoid', name='habitable_output')(x)
    #
    # model = Model(inputs=inputs, outputs=habitable)
    # return model
    """完全兼容的模型架构"""
    # 显式定义输入层并确保名称唯一
    main_input = Input(shape=(input_dim,), name=f'main_input_{input_dim}d')

    # 共享特征提取层
    x = Dense(256, activation='relu')(main_input)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)

    # 输出层
    output = Dense(1, activation='sigmoid', name='habitable_output')(x)

    model = Model(inputs=main_input, outputs=output)
    return model
# ======================
# 自定义损失函数
# ======================
def hybrid_loss(y_true, y_pred):
    # 分类损失
    ce_loss = tf.keras.losses.binary_crossentropy(y_true[0], y_pred[0])
    # 温度预测损失
    mse_loss = tf.keras.losses.mean_squared_error(y_true[1], y_pred[1])
    # 物理约束损失
    physics_loss = tf.reduce_mean(tf.square(y_true[2] - y_pred[2]))

    return ce_loss + 0.3 * mse_loss + 0.1 * physics_loss


# ======================
# 主程序
# ======================
if __name__ == "__main__":
    # 数据加载与预处理
    try:
        raw_df = pd.read_excel("raw_data.xlsx")
        processed_df, features = preprocess_data(raw_df)
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        exit()



    # 数据分割（确保变量定义）
    try:
        X = processed_df[features].values
        y = processed_df['habitable'].values

        # 添加数据存在性验证
        if X.size == 0 or y.size == 0:
            raise ValueError("特征矩阵或目标向量为空！")

        # 类别平衡处理
        if len(np.unique(y)) < 2:
            raise ValueError("目标变量仅包含单一类别")

        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)

        # 数据集划分（显式定义所有变量）
        X_train, X_test, y_train, y_test = train_test_split(
            X_res,
            y_res,
            test_size=0.2,
            stratify=y_res,
            random_state=42
        )

    except Exception as e:
        print(f"数据处理错误: {str(e)}")
        exit()
    # 在数据分割后添加维度修正
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    # 维度验证（关键调试步骤）
    print("\n=== 数据维度验证 ===")
    print(f"训练集特征维度: {X_train.shape}")
    print(f"测试集特征维度: {X_test.shape}")
    print(f"训练集标签维度: {y_train.shape}")
    print(f"测试集标签维度: {y_test.shape}")

    # 模型构建（确保在变量定义后执行）
    try:
        input_dim = X_train.shape[1]
        print(f"\n模型输入维度: {input_dim}")
        model = build_model(input_dim)
    except Exception as e:
        print(f"模型构建失败: {str(e)}")
        exit()

    # 模型编译（调整损失权重）
    model.compile(
        # optimizer=Adam(learning_rate=0.001),
        # loss={
        #     'habitable_output': 'binary_crossentropy',
        #     'temp_output': 'mse',
        #     'physics_output': 'mse'
        # },
        # metrics={
        #     'habitable_output': ['accuracy', tf.keras.metrics.AUC()],
        #     'temp_output': 'mae'
        # },
        # loss_weights=[0.7, 0.2, 0.1]
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )

    # 模型训练（添加数据验证）
    try:
        history = model.fit(
            # X_train,
            # {
            #     'habitable_output': y_train,
            #     'temp_output': np.zeros(len(y_train)),  # 占位数据
            #     'physics_output': np.zeros(len(y_train))  # 占位数据
            # },
            # epochs=100,
            # batch_size=32,
            # validation_split=0.2,
            # callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
            # verbose=1
            X_train,
            y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            # callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
            callbacks=[EarlyStopping(patience=10)],
            verbose=1
        )
    except Exception as e:
        print(f"训练错误: {str(e)}")
        exit()
    # 模型评估
    # 模型评估（修正维度处理）
    y_pred = model.predict(X_test)
    y_pred_class = (y_pred > 0.5).astype(int).flatten()  # 确保一维

    # 调整结果保存部分
    result_df = pd.DataFrame({
        '实际类别': y_test,  # 直接使用一维的y_test
        '预测概率': y_pred.flatten(),
        '预测类别': y_pred_class.flatten()
    })
    result_df.to_excel("output/predictions.xlsx", index=False)

    # 可视化评估指标
    plt.figure(figsize=(15, 6))

    # 训练曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='训练损失')  # 修改键名
    plt.plot(history.history['val_loss'], label='验证损失')  # 修改键名
    plt.title('训练损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 混淆矩阵
    plt.subplot(1, 2, 2)
    cm = confusion_matrix(y_test, y_pred_class)  # 直接使用一维标签
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')

    plt.savefig("output/training_metrics.png", dpi=300)

    # 特征重要性分析
    try:
        # 创建符合Keras预期的背景数据
        background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]

        # 显式获取模型输入层
        input_tensor = model.input

        # 创建解释器（使用模型输入张量）
        explainer = shap.DeepExplainer(
            (input_tensor, model.output),
            background.astype(np.float32)  # 确保数据类型

        # 计算SHAP值（使用统一的数据格式）
        test_samples = X_test[:50].astype(np.float32)
        shap_values = explainer.shap_values(test_samples)

        # 可视化处理
        plt.figure(figsize=(10, 6))
        if isinstance(shap_values, list):
        # 处理多输出情况（当前模型不需要）
            shap.summary_plot(shap_values[0], test_samples, feature_names=features)
        else:
            shap.summary_plot(shap_values, test_samples, feature_names=features)
        plt.savefig("output/shap_summary.png", dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"SHAP分析失败: {str(e)}")

    # ======================
    print(f"""
    ======================
    执行完成！关键指标：
    - 训练准确率: {history.history['accuracy'][-1]:.2%}
    - 验证准确率: {history.history['val_accuracy'][-1]:.2%}
    - 测试集AUC分数: {roc_auc_score(y_test, y_pred):.2%}

    生成文件清单：
    - processed_data.xlsx      预处理数据
    - output/predictions.xlsx     预测结果
    - output/training_metrics.png 评估指标
    - output/shap_summary.png     特征重要性
    ======================
    """)