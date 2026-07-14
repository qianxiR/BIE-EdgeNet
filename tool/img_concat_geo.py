import os
import numpy as np
import cv2
import osgeo.gdal as gdal
import osgeo.osr as osr
import pandas as pd
import rasterio  # 用于地理→像素坐标转换

# -------------------------- 【必改参数】请根据你的实际路径修改 --------------------------
# 1. 预测图斑文件夹（SOTA模型输出的png图斑，如test_1.png）
PNG_DIR = r"E:\zyh\Data\CDCD\CDCD_test\CDCDrepair-CDCD-9-1-10\pred_result"
# 2. 地理参考CSV路径（你当前的test_geo_coords.csv，含file_name/geo_lon/geo_lat）
GEO_CSV_PATH = r"E:\zyh\Data\CDCD\CDCD_test\geo_info\test_geo_coords.csv"
# 3. 原始影像路径（用于读取正确的地理参考，如zyhCD20160607.tif）
ORIGINAL_IMG_PATH = r"E:\CDCD\zyhCD20160607\Level18\zyhCD20160607.tif"
# 4. 最终拼接结果输出路径（带地理参考的TIFF，如concat_CDCD_pre.tif）
OUTPUT_TIF_PATH = r"E:\zyh\Data\CDCD\CDCD_test\CDCDrepair-CDCD-9-1-10\concat_CDCD_pre.tif"
# -----------------------------------------------------------------------------------

def main():
    #  Step 1：读取原始影像的地理信息（核心！复用正确的transform/CRS/尺寸）
    print("✅ 读取原始影像地理信息...")
    with rasterio.open(ORIGINAL_IMG_PATH) as src:
        # 原始影像的地理变换参数（像素→地理坐标映射，直接复用）
        original_transform = src.transform
        # 原始影像的坐标系（CRS，直接复用，避免手动设置EPSG错误）
        original_crs = src.crs
        # 原始影像的像素尺寸（宽/高，拼接后的影像尺寸与原始一致）
        original_width = src.width
        original_height = src.height
        # 原始影像的像素分辨率（可选，用于验证）
        pixel_res_x = src.res[0]
        pixel_res_y = src.res[1]
    print(f"📌 原始影像信息：")
    print(f"   - 尺寸：{original_width}px × {original_height}px")
    print(f"   - 分辨率：{pixel_res_x}（x方向）, {pixel_res_y}（y方向）")
    print(f"   - 坐标系：{original_crs}")

    #  Step 2：读取地理参考CSV（你的test_geo_coords.csv）
    print("\n✅ 读取地理参考CSV...")
    geo_df = pd.read_csv(GEO_CSV_PATH)
    # 验证CSV必要列是否存在（避免列名错误）
    required_cols = ["file_name", "geo_lon", "geo_lat", "tile_size"]
    for col in required_cols:
        if col not in geo_df.columns:
            raise ValueError(f"❌ CSV缺少必要列：{col}，请检查CSV结构")
    # 获取瓦片尺寸（默认所有瓦片尺寸一致）
    tile_size = int(geo_df["tile_size"].iloc[0])
    print(f"📌 CSV信息：共{len(geo_df)}个瓦片，瓦片尺寸：{tile_size}×{tile_size}px")

    #  Step 3：初始化完整拼接影像（尺寸与原始影像完全一致，背景值0）
    print("\n✅ 初始化完整拼接影像...")
    full_mask = np.zeros((original_height, original_width), dtype=np.uint8)  # 0=背景，255=图斑

    #  Step 4：逐瓦片读取+地理坐标→像素坐标转换+填充到完整影像
    print("\n✅ 开始拼接图斑（共{len(geo_df)}个瓦片）...")
    missing_count = 0  # 统计缺失/损坏的瓦片
    for idx, row in geo_df.iterrows():
        # 4.1 获取当前瓦片信息
        tif_name = row["file_name"]  # CSV中的TIFF文件名（如test_1.tif）
        png_name = tif_name.replace(".tif", ".png")  # 转为PNG文件名（如test_1.png）
        png_path = os.path.join(PNG_DIR, png_name)
        tile_geo_lon = row["geo_lon"]  # 瓦片左上角经度
        tile_geo_lat = row["geo_lat"]  # 瓦片左上角纬度

        # 4.2 检查PNG文件是否存在
        if not os.path.exists(png_path):
            print(f"⚠️  瓦片{idx+1}/{len(geo_df)}：{png_name} 不存在，跳过")
            missing_count += 1
            continue

        # 4.3 读取PNG图斑（单波段灰度图，0=背景，255=图斑）
        png_mask = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        # 检查PNG是否读取成功+尺寸是否匹配
        if png_mask is None:
            print(f"⚠️  瓦片{idx+1}/{len(geo_df)}：{png_name} 读取失败，跳过")
            missing_count += 1
            continue
        if png_mask.shape != (tile_size, tile_size):
            print(f"⚠️  瓦片{idx+1}/{len(geo_df)}：{png_name} 尺寸{png_mask.shape}≠{tile_size}×{tile_size}，跳过")
            missing_count += 1
            continue

        # 4.4 地理坐标→像素坐标（关键！用原始影像的transform转换）
        # rasterio.transform.rowcol(transform, 经度, 纬度) → 返回（行号y，列号x）
        tile_y, tile_x = rasterio.transform.rowcol(
            transform=original_transform,
            xs=tile_geo_lon,
            ys=tile_geo_lat
        )
        # 转为整数（像素坐标必须是整数）
        tile_x = int(tile_x)
        tile_y = int(tile_y)

        # 4.5 验证瓦片是否在原始影像范围内（避免越界）
        if (tile_x + tile_size > original_width) or (tile_y + tile_size > original_height):
            print(f"⚠️  瓦片{idx+1}/{len(geo_df)}：{png_name} 超出原始影像范围，跳过")
            missing_count += 1
            continue

        # 4.6 将当前瓦片填充到完整影像的对应位置
        full_mask[tile_y:tile_y+tile_size, tile_x:tile_x+tile_size] = png_mask
        # 进度提示（每10个瓦片打印一次）
        if (idx + 1) % 10 == 0 or (idx + 1) == len(geo_df):
            print(f"🔄 已处理{idx+1}/{len(geo_df)}个瓦片，缺失{missing_count}个")

    #  Step 5：写入带地理参考的TIFF（完全复用原始影像的地理参数）
    print("\n✅ 写入带地理参考的TIFF...")
    # 5.1 创建GDAL驱动（GTiff格式）
    driver = gdal.GetDriverByName("GTiff")
    if driver is None:
        raise ValueError("❌ 无法创建GDAL GTiff驱动，请检查GDAL安装")

    # 5.2 创建TIFF文件（尺寸、波段数、数据类型与原始影像匹配）
    dataset = driver.Create(
        utf8_path=OUTPUT_TIF_PATH,  # GDAL 3.x要求用utf8_path指定路径
        xsize=original_width,       # 宽度=原始影像宽度
        ysize=original_height,      # 高度=原始影像高度
        bands=1,                    # 波段数=1（单波段二值图）
        eType=gdal.GDT_Byte         # 数据类型=8位无符号整数（0-255）
    )
    if dataset is None:
        raise ValueError(f"❌ 无法创建TIFF文件：{OUTPUT_TIF_PATH}，请检查路径权限")

    # 5.3 设置地理变换参数（完全复用原始影像的transform）
    dataset.SetGeoTransform((
        original_transform[0],  # 左上角x坐标（经度）
        original_transform[1],  # x方向像素分辨率
        0.0,                    # 旋转参数（无旋转）
        original_transform[3],  # 左上角y坐标（纬度）
        0.0,                    # 旋转参数（无旋转）
        original_transform[5]   # y方向像素分辨率（通常为负）
    ))

    # 5.4 设置坐标系（完全复用原始影像的CRS，避免手动设置错误）
    srs = osr.SpatialReference()
    srs.ImportFromWkt(original_crs.to_wkt())  # 用原始影像的WKT格式CRS
    dataset.SetProjection(srs.ExportToWkt())

    # 5.5 写入拼接后的图斑数据
    dataset.GetRasterBand(1).WriteArray(full_mask)
    # 设置NoData值（可选，空白区域设为0，与背景一致）
    dataset.GetRasterBand(1).SetNoDataValue(0)
    # 刷新缓存+释放资源（避免文件占用）
    dataset.FlushCache()
    dataset = None

    #  Step 6：拼接完成提示+验证建议
    print("\n" + "="*60)
    print("🎉 图斑拼接完成！")
    print(f"📁 输出文件路径：{OUTPUT_TIF_PATH}")
    print(f"📊 拼接统计：共{len(geo_df)}个瓦片，成功{len(geo_df)-missing_count}个，缺失/损坏{missing_count}个")
    print(f"🌍 地理参考：与原始影像完全一致（{original_crs}）")
    print("\n🔍 ArcGIS验证步骤：")
    print("   1. 导入拼接后的TIFF文件")
    print("   2. 右键TIFF→「缩放至图层」（自动定位到原始影像范围）")
    print("   3. 右键TIFF→「符号系统」→「唯一值」：")
    print("      - 255 → 设置红色（图斑）")
    print("      - 0 → 设置透明（背景）")
    print("   4. 叠加原始影像，确认图斑位置匹配")
    print("="*60)

if __name__ == "__main__":
    main()