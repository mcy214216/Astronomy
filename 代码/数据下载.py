import pandas as pd
import requests

# 下载 TESS TOI 列表
url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

try:
    print("正在下载数据...")
    response = requests.get(url, headers=headers, timeout=30)
    response.encoding = 'utf-8'  # 或 'latin1' 如果出现编码错误
    df = pd.read_csv(pd.compat.StringIO(response.text))

    # 保存为 Excel
    df.to_excel("tess_raw_data.xlsx", index=False, engine='openpyxl')
    # 同时保存 CSV 备份
    df.to_csv("tess_raw_backup.csv", index=False, encoding='utf-8-sig')

    print(f"下载完成！数据维度: {df.shape}，已保存为 tess_raw_data.xlsx")
except Exception as e:
    print(f"下载失败: {e}")