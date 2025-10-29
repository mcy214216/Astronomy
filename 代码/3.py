import pandas as pd

# 读取原始数据
data_path = "Pantheon+SH0ES.dat"
df = pd.read_csv(
    data_path,
    sep=r"\s+",          # 匹配任意空白符
    # skiprows=1,          # 跳过首行注释
    header=0,            # 列名在第2行
    engine="python",     # 确保兼容正则表达式
    na_values=["-9"]     # 将 -9 标记为缺失值
)

# 保存为 Excel
excel_path = "Pantheon+SH0ES.xlsx"
df.to_excel(excel_path, index=False)