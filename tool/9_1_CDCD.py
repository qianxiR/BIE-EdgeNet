import os
import cv2
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from os.path import join as ospj
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box

# ====== 配置参数（严格保留你训练时的9:1划分） ======
tile_size = 256
stride = 256
split_ratio = [0.9, 0.1, 0.0]  # 和你训练SOTA模型一致的9:1划分
random_seed = 42  # 固定种子，保证划分和训练时完全一致

# 输入路径（和你训练时完全一致）
img_A_path = r"E:\CDCD\zyhCD20160607\Level18\zyhCD20160607.tif"
img_B_path = r"E:\CDCD\zyhCD20230308\Level18\zyhCD20230308.tif"
label_path = r"E:\CDCD\CDCDLabel.tif"
net_vector_path = r"E:\CDCD\CDCDNet.shp"  # 原始CDCDNet渔网（要添加属性的核心）

# 输出路径（测试集包含所有训练/验证数据，保存test_1~test_N）
save_root = r'E:\zyh\Data\CDCD\CDCD_test'
# 保持和训练时一致的目录结构
os.makedirs(ospj(save_root, 'A'), exist_ok=True)
os.makedirs(ospj(save_root, 'B'), exist_ok=True)
os.makedirs(ospj(save_root, 'label'), exist_ok=True)
os.makedirs(ospj(save_root, 'A_tif'), exist_ok=True)
os.makedirs(ospj(save_root, 'B_tif'), exist_ok=True)
os.makedirs(ospj(save_root, 'label_tif'), exist_ok=True)
os.makedirs(ospj(save_root, 'list'), exist_ok=True)
os.makedirs(ospj(save_root, 'geo_info'), exist_ok=True)
os.makedirs(ospj(save_root, 'vector'), exist_ok=True)  # 保存带属性的完整CDCDNet


# ====== 工具函数（完全复用你训练时的逻辑） ======
def slide_crop(img, size=1024, stride=1024):
    h, w = img.shape[:2]
    crops = []
    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
            crops.append((x, y))
    return crops


def save_crop(img, x, y, size, path):
    crop = img[y:y + size, x:x + size]
    cv2.imwrite(path, crop)


def save_crop_with_geo(img, x, y, size, path, src_transform):
    crop_transform = from_origin(
        west=src_transform[0] + x * src_transform[1],
        north=src_transform[3] + y * src_transform[5],
        xsize=src_transform[1],
        ysize=abs(src_transform[5])
    )
    if len(img.shape) == 3:
        count = 3
        dtype = np.uint8
    else:
        count = 1
        dtype = np.uint8
    with rasterio.open(
        path, 'w',
        driver='GTiff',
        height=size,
        width=size,
        count=count,
        dtype=dtype,
        crs=rasterio.open(img_A_path).crs,
        transform=crop_transform,
    ) as dst:
        if count == 3:
            dst.write(img[y:y + size, x:x + size].transpose(2, 0, 1))
        else:
            dst.write(img[y:y + size, x:x + size], 1)


def vector_to_mask(raster_path, vector_path):
    with rasterio.open(raster_path) as src:
        transform = src.transform
        width = src.width
        height = src.height
        crs = src.crs
    gdf = gpd.read_file(vector_path)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    shapes = [(geom, 255) for geom in gdf.geometry if not geom.is_empty]
    mask = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )
    return mask


def is_valid_crop(net_mask, x, y, tile_size, threshold=0.1):
    mask_crop = net_mask[y:y + tile_size, x:x + tile_size]
    valid_pixel_ratio = np.count_nonzero(mask_crop == 255) / (tile_size * tile_size)
    return valid_pixel_ratio > threshold


def cal_geo_coords(x, y, src_transform):
    lon = src_transform[0] + x * src_transform[1]
    lat = src_transform[3] + y * src_transform[5]
    return round(lon, 6), round(lat, 6)


# ====== 主流程：9:1划分+全量生成test名称+CDCDNet添加属性 ======
def main():
    # 1. 读取原始影像和地理信息（和训练时一致）
    img_A = cv2.imread(img_A_path)
    img_B = cv2.imread(img_B_path)
    label = cv2.imread(label_path, 0)
    with rasterio.open(img_A_path) as src:
        src_transform = src.transform
        src_crs = src.crs
        src_width = src.width
        src_height = src.height

    # 2. 读取原始CDCDNet渔网（完整网格，后续要给每个网格加属性）
    print("📌 读取原始CDCDNet完整渔网...")
    cdcdnet_gdf = gpd.read_file(net_vector_path)
    if cdcdnet_gdf.crs != src_crs:
        cdcdnet_gdf = cdcdnet_gdf.to_crs(src_crs)
    total_grids = len(cdcdnet_gdf)
    print(f"📌 CDCDNet总网格数：{total_grids}")

    # 3. 严格按训练时的逻辑划分train/val（保证划分一致）
    random.seed(random_seed)
    cdcdnet_shuffled = cdcdnet_gdf.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    num_train = int(split_ratio[0] * total_grids)
    num_val = total_grids - num_train
    splits = {
        'train': cdcdnet_shuffled.iloc[:num_train],
        'val': cdcdnet_shuffled.iloc[num_train:]
    }
    print(f"✅ 按9:1划分（和训练一致）：训练集{num_train}个，验证集{num_val}个")

    # 4. 初始化变量：全量测试集名称（test_1~test_N）、地理信息、矢量属性
    test_global_idx = 1  # 全局测试集序号（连续，覆盖train+val）
    all_test_names = []  # 每个CDCDNet网格对应的test名称（按shuffled顺序）
    all_test_geo = []    # 所有测试集的地理信息
    test_list = []       # 测试集list文件
    geo_cols = ['file_name', 'geo_lon', 'geo_lat', 'geo_minx', 'geo_miny', 'geo_maxx', 'geo_maxy', 'tile_size', 'crs']

    # 5. 先处理训练集→转为test_xxx
    print("\n🔄 处理训练集（转为测试集test_xxx）...")
    for idx, row in tqdm(splits['train'].iterrows(), total=len(splits['train']), desc='训练集→测试集'):
        minx, miny, maxx, maxy = row.geometry.bounds
        x, y = rasterio.transform.rowcol(src_transform, minx, maxy)
        x = int(x)
        y = int(y)

        # 跳过越界网格
        if x + tile_size > src_width or y + tile_size > src_height:
            all_test_names.append(None)  # 越界网格标记为None
            continue

        # 生成连续的测试集名称（test_1, test_2...）
        test_name = f"test_{test_global_idx}"
        all_test_names.append(test_name)

        # 保存PNG/TIFF（和训练时格式一致，但命名为test_xxx）
        save_crop(img_A, x, y, tile_size, ospj(save_root, 'A', test_name + '.png'))
        save_crop(img_B, x, y, tile_size, ospj(save_root, 'B', test_name + '.png'))
        save_crop(label, x, y, tile_size, ospj(save_root, 'label', test_name + '.png'))
        save_crop_with_geo(img_A, x, y, tile_size, ospj(save_root, 'A_tif', test_name + '.tif'), src_transform)
        save_crop_with_geo(img_B, x, y, tile_size, ospj(save_root, 'B_tif', test_name + '.tif'), src_transform)
        save_crop_with_geo(label, x, y, tile_size, ospj(save_root, 'label_tif', test_name + '.tif'), src_transform)

        # 记录地理信息
        all_test_geo.append([
            test_name + '.tif',
            minx, maxy,
            minx, miny, maxx, maxy,
            tile_size,
            str(src_crs)
        ])
        test_list.append(test_name + '.png\n')
        test_global_idx += 1

    # 6. 再处理验证集→转为test_xxx（序号连续）
    print("\n🔄 处理验证集（转为测试集test_xxx）...")
    for idx, row in tqdm(splits['val'].iterrows(), total=len(splits['val']), desc='验证集→测试集'):
        minx, miny, maxx, maxy = row.geometry.bounds
        x, y = rasterio.transform.rowcol(src_transform, minx, maxy)
        x = int(x)
        y = int(y)

        # 跳过越界网格
        if x + tile_size > src_width or y + tile_size > src_height:
            all_test_names.append(None)  # 越界网格标记为None
            continue

        # 生成连续的测试集名称（接在训练集后面）
        test_name = f"test_{test_global_idx}"
        all_test_names.append(test_name)

        # 保存PNG/TIFF
        save_crop(img_A, x, y, tile_size, ospj(save_root, 'A', test_name + '.png'))
        save_crop(img_B, x, y, tile_size, ospj(save_root, 'B', test_name + '.png'))
        save_crop(label, x, y, tile_size, ospj(save_root, 'label', test_name + '.png'))
        save_crop_with_geo(img_A, x, y, tile_size, ospj(save_root, 'A_tif', test_name + '.tif'), src_transform)
        save_crop_with_geo(img_B, x, y, tile_size, ospj(save_root, 'B_tif', test_name + '.tif'), src_transform)
        save_crop_with_geo(label, x, y, tile_size, ospj(save_root, 'label_tif', test_name + '.tif'), src_transform)

        # 记录地理信息
        all_test_geo.append([
            test_name + '.tif',
            minx, maxy,
            minx, miny, maxx, maxy,
            tile_size,
            str(src_crs)
        ])
        test_list.append(test_name + '.png\n')
        test_global_idx += 1

    # 7. 核心操作：给原始CDCDNet完整渔网添加sample_name属性
    print("\n✅ 给CDCDNet添加测试集名称属性...")
    # 给shuffled后的CDCDNet添加属性（和划分顺序一致）
    cdcdnet_shuffled['sample_name'] = all_test_names
    # 恢复原始CDCDNet的索引（保证和原始渔网顺序一致）
    cdcdnet_final = cdcdnet_shuffled.sort_index().reset_index(drop=True)
    # 保存带属性的完整CDCDNet（每个网格都有对应的test_xxx，越界为None）
    final_vector_path = ospj(save_root, 'vector', 'CDCDNet_With_TestName.shp')
    cdcdnet_final.to_file(final_vector_path, driver='ESRI Shapefile', encoding='utf-8')
    print(f"✅ 带测试集名称的CDCDNet已保存：{final_vector_path}")

    # 8. 保存测试集地理信息CSV（仅有效网格）
    test_geo_df = pd.DataFrame(all_test_geo, columns=geo_cols)
    geo_csv_path = ospj(save_root, 'geo_info', 'test_geo_coords.csv')
    test_geo_df.to_csv(geo_csv_path, index=False, encoding='utf-8')
    print(f"✅ 测试集地理坐标表已保存：{geo_csv_path}")

    # 9. 保存测试集list文件（供模型预测加载）
    with open(ospj(save_root, 'list', 'test.txt'), 'w') as f:
        f.writelines(test_list)

    # 最终统计
    valid_test_num = test_global_idx - 1
    print('\n' + '='*70)
    print(f'✅ 全流程完成！核心结果：')
    print(f'   1. 测试集覆盖所有训练/验证数据：共{valid_test_num}个有效样本（test_1~test_{valid_test_num}）')
    print(f'   2. CDCDNet属性：{final_vector_path} → 每个网格的sample_name对应测试集名称')
    print(f'   3. 预测结果对照：找到sample_name=test_N → 对应预测结果test_N.png → 定位原始图斑位置')
    print(f'   4. 越界网格：sample_name为None（共{total_grids - valid_test_num}个）')
    print('='*70)


if __name__ == '__main__':
    main()