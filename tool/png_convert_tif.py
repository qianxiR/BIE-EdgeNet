import os
import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from tqdm import tqdm
from os.path import join as ospj

# ====== 仅需修改这3个路径！！！ ======
pred_png_root = r"E:\zyh\Data\CDCD\CDCD_test\pred"  # 模型预测的PNG保存根目录（所有test_xx.png都在这）
geo_csv_path = r"E:\zyh\Data\CDCD\CDCD_test\geo_info\test_geo_coords.csv"  # 之前生成的地理坐标表
save_tif_root = r"E:\zyh\Data\CDCD\CDCD_test\pred_tif"  # 转换后的地理化TIFF保存根目录
# ====================================

# 创建TIFF保存目录
os.makedirs(save_tif_root, exist_ok=True)

# 读取地理坐标表
geo_df = pd.read_csv(geo_csv_path, encoding='utf-8')
# 读取原始影像的像素尺度（与预测图一致，从原img_A读取，保证精度）
src_raster_path = r"E:\CDCD\zyhCD20160607\Level18\zyhCD20160607.tif"  # 原img_A路径，无需修改
with rasterio.open(src_raster_path) as src:
    pixel_xsize = src.transform[1]  # 像素宽度（地理单位/像素）
    pixel_ysize = abs(src.transform[5])  # 像素高度（取绝对值）
    src_crs = src.crs  # 坐标系，与原影像一致

# 批量转换：PNG → 带地理信息TIFF
print(f"开始转换，共{len(geo_df)}个预测PNG...")
for idx, row in tqdm(geo_df.iterrows(), total=len(geo_df)):
    # 获取每个test_xx的信息
    file_name = row['file_name']  # test_x.tif（用于匹配PNG）
    test_name = file_name.replace('.tif', '')  # 提取test_x，匹配预测PNG
    pixel_x = row['pixel_x']
    pixel_y = row['pixel_y']
    geo_lon = row['geo_lon']  # 左上角经度
    geo_lat = row['geo_lat']  # 左上角纬度

    # 读取预测PNG（二值化变化图，单通道）
    pred_png_path = ospj(pred_png_root, f"{test_name}.png")
    if not os.path.exists(pred_png_path):
        print(f"⚠️  未找到预测PNG：{pred_png_path}，跳过")
        continue
    pred_img = cv2.imread(pred_png_path, 0)  # 单通道读取（变化检测预测图为二值化，0/255或0/1）
    tile_size = pred_img.shape[0]  # 256，与原尺寸一致

    # 计算该预测图的地理变换参数（与原Test集TIFF完全一致）
    pred_transform = from_origin(
        west=geo_lon,
        north=geo_lat,
        xsize=pixel_xsize,
        ysize=pixel_ysize
    )

    # 保存为带地理信息的TIFF
    pred_tif_path = ospj(save_tif_root, f"{test_name}.tif")
    with rasterio.open(
        pred_tif_path, 'w',
        driver='GTiff',
        height=tile_size,
        width=tile_size,
        count=1,  # 预测图为单通道（二值化）
        dtype=np.uint8,
        crs=src_crs,
        transform=pred_transform,
    ) as dst:
        dst.write(pred_img, 1)  # 写入单通道

print(f"✅ 转换完成！地理化预测TIFF保存在：{save_tif_root}")
print(f"✅ 可直接在ArcGIS中使用「镶嵌至新栅格」工具拼接该目录下的所有TIFF")