import pandas as pd

# 仅保留「读取CSV+打印列名」的代码，其他都注释掉
geo_ref_path = r"E:\zyh\Data\CDCD\CDCD_test\geo_info\test_geo_coords.csv"  # 你的CSV路径，确保正确

# 1. 读取CSV
geo_ref = pd.read_csv(geo_ref_path)

# 2. 打印CSV的「真实列名」（关键！看清楚到底是什么列名）
print("="*50)
print("你的CSV文件实际列名：")
print(geo_ref.columns.tolist())  # 打印所有列名，比如可能是['row', 'col']、[' Row', ' Col']（带空格）等
print("="*50)
print("CSV前5行数据预览（确认列名对应的数据）：")
print(geo_ref.head())