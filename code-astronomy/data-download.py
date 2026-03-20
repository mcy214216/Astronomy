# -*- coding: utf-8 -*-
# 功能：从TESS官方数据源下载候选行星数据，保存为本地Excel文件

import requests
import pandas as pd
import time
from pathlib import Path

# ======================
# 配置参数
# ======================
TESS_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
OUTPUT_FILE = "tess_raw_data.xlsx"
REQUEST_TIMEOUT = 60  # 请求超时时间（秒）
MAX_RETRIES = 3  # 最大重试次数
RETRY_DELAY = 5  # 重试间隔（秒）

# 原始列名映射（保持与原代码一致）
COLUMN_MAP = {
    'Period (days)': 'period',
    'Duration (hours)': 'duration',
    'Depth (ppm)': 'depth',
    'Planet Radius (R_Earth)': 'radius',
    'Equilibrium Temperature (K)': 'temp',
    'Stellar Radius (R_Sun)': 'stellar_radius',
    'Planet SNR': 'snr',
    'TFOPWG Disposition': 'disposition',
    'Comments': 'comments'
}


# ======================
# 下载函数
# ======================
def download_tess_data():
    """
    从TESS网站下载候选行星数据，返回DataFrame
    若失败则返回None
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"正在尝试下载数据 (第{attempt}次)...")
            response = requests.get(TESS_URL, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()  # 检查HTTP错误

            # 将CSV内容读取为DataFrame
            from io import StringIO
            data = pd.read_csv(StringIO(response.text))
            print(f"下载成功！共获取 {len(data)} 条记录。")
            return data

        except requests.exceptions.RequestException as e:
            print(f"第{attempt}次下载失败: {e}")
            if attempt < MAX_RETRIES:
                print(f"等待 {RETRY_DELAY} 秒后重试...")
                time.sleep(RETRY_DELAY)
            else:
                print("已达到最大重试次数，下载失败。")
                return None
        except Exception as e:
            print(f"数据处理时出错: {e}")
            return None


def save_to_excel(df, filename):
    """将DataFrame保存为Excel文件"""
    try:
        # 确保目录存在（如果需要）
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(filename, index=False, engine='openpyxl')
        print(f"数据已保存至: {filename}")
    except Exception as e:
        print(f"保存Excel文件失败: {e}")


def main():
    """主流程"""
    print("=== TESS数据下载工具 ===")

    # 下载数据
    df = download_tess_data()
    if df is None:
        print("程序终止：无法获取数据。")
        return

    # 可选：查看基本信息
    print("\n原始数据概览：")
    print(f"  行数: {df.shape[0]}")
    print(f"  列数: {df.shape[1]}")
    print(f"  列名: {list(df.columns)[:10]}...")  # 显示前10列

    # 保存原始数据（保留原始列名）
    save_to_excel(df, OUTPUT_FILE)

    # 额外：如果用户需要，可以同时保存一份带有重命名列的版本（用于直接集成）
    # 可根据需要注释掉
    try:
        df_renamed = df.rename(columns=COLUMN_MAP)
        renamed_file = OUTPUT_FILE.replace(".xlsx", "_renamed.xlsx")
        save_to_excel(df_renamed, renamed_file)
        print(f"同时保存了重命名列版本: {renamed_file}")
    except Exception as e:
        print(f"保存重命名版本失败: {e}")

    print("\n下载完成！")


if __name__ == "__main__":
    main()