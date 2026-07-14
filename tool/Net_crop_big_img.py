import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from os.path import join as ospj
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box

# ====== 配置参数 ======
tile_size = 256
stride = 256
split_ratio = [0.7, 0.1, 0.2]
random_seed = 42

# 输入路径
img_A_path = r"E:\CDCD\zyhCD20160607\Level18\zyhCD20160607.tif"
img_B_path = r"E:\CDCD\zyhCD20230308\Level18\zyhCD20230308.tif"
label_path = r"E:\CDCD\CDCDcheckLabel.tif"
# ========== 关键修改：渔网矢量文件路径（SHP格式） ==========
net_vector_path = r"E:\CDCD\CDCDNet.shp"  # 替换为你的渔网SHP路径

# 输出路径
save_root = r'E:\zyh\Data\CDCD\CDCD_check_712'
os.makedirs(ospj(save_root, 'A'), exist_ok=True)
os.makedirs(ospj(save_root, 'B'), exist_ok=True)
os.makedirs(ospj(save_root, 'label'), exist_ok=True)
os.makedirs(ospj(save_root, 'list'), exist_ok=True)


# ====== 工具函数 ======
def slide_crop(img, size=1024, stride=1024):
    h, w = img.shape[:2]
    crops = []
    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
            crops.append((x, y))
    return crops


# def save_crop(img, x, y, size, path):
#     crop = img[y:y + size, x:x + size]
#     cv2.imwrite(path, crop)
def save_crop(img, x, y, size, path):
    crop = img[y:y + size, x:x + size]
    # 如果是标签数据（0/1），则转换为 0/255
    if crop.dtype == np.uint8 and crop.max() <= 1:
        crop = (crop * 255).astype(np.uint8)
    cv2.imwrite(path, crop)


# ========== 新增：矢量渔网转图像掩膜 ==========
def vector_to_mask(raster_path, vector_path):
    """
    将渔网矢量文件转换为和栅格影像尺寸、地理范围一致的掩膜
    Args:
        raster_path: 参考栅格路径（你的img_A路径）
        vector_path: 渔网矢量SHP路径
    Returns:
        mask: 二维数组，有效区域（渔网覆盖）为255，无效区域为0
    """
    # 读取参考栅格的地理信息
    with rasterio.open(raster_path) as src:
        transform = src.transform  # 地理变换参数
        width = src.width  # 栅格宽度
        height = src.height  # 栅格高度
        crs = src.crs  # 坐标系统

    # 读取渔网矢量文件
    gdf = gpd.read_file(vector_path)
    # 确保矢量和栅格坐标系统一致（不一致则投影转换）
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    # 生成掩膜：渔网覆盖区域设为255，无覆盖设为0
    shapes = [(geom, 255) for geom in gdf.geometry if not geom.is_empty]
    mask = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,  # 无渔网覆盖的区域填充0
        dtype=np.uint8
    )
    return mask


# ========== 检查裁剪块是否在渔网有效区域内 ==========
def is_valid_crop(net_mask, x, y, tile_size, threshold=0.1):
    """
    检查裁剪块是否在渔网有效区域内
    Args:
        net_mask: 矢量转换后的掩膜（有效=255，无效=0）
        x, y: 裁剪块左上角坐标
        tile_size: 裁剪块尺寸
        threshold: 有效像素占比阈值（大于该值则保留）
    Returns:
        bool: True=有效，False=无效
    """
    # 提取裁剪块对应的掩膜区域
    mask_crop = net_mask[y:y + tile_size, x:x + tile_size]
    # 计算有效像素（255）占比
    valid_pixel_ratio = np.count_nonzero(mask_crop == 255) / (tile_size * tile_size)
    # 有效像素占比大于阈值则保留该块
    return valid_pixel_ratio > threshold


# ====== 主处理流程 ======
def main():
    # 读取图像
    img_A = cv2.imread(img_A_path)
    img_B = cv2.imread(img_B_path)
    # label = cv2.imread(label_path, 0)  # 单通道读取
    with rasterio.open(label_path) as src:
        label = src.read(1)  # 读取第一个波段，根据实际情况调整
        # 如果是多波段，可使用 src.read() 读取所有波段

    # ========== 关键步骤：将渔网矢量转为掩膜 ==========
    print("正在将渔网矢量转换为图像掩膜...")
    net_mask = vector_to_mask(img_A_path, net_vector_path)
    # 校验尺寸一致性
    assert img_A.shape[:2] == img_B.shape[:2] == label.shape[:2] == net_mask.shape[:2], \
        "图像、标签、渔网掩膜的尺寸不一致！"

    # 生成所有可能的裁剪坐标
    all_coords = slide_crop(img_A, size=tile_size, stride=stride)
    # 筛选渔网有效区域内的坐标
    valid_coords = []
    print(f"正在筛选渔网有效区域内的裁剪块...")
    for (x, y) in tqdm(all_coords, desc="筛选有效块"):
        if is_valid_crop(net_mask, x, y, tile_size):
            valid_coords.append((x, y))

    print(f"原始裁剪块总数: {len(all_coords)}, 渔网有效块数: {len(valid_coords)}")
    if len(valid_coords) == 0:
        print("❌ 没有找到渔网有效区域内的裁剪块，请检查矢量文件或路径！")
        return

    # 随机打乱有效坐标（保持原有随机种子）
    random.seed(random_seed)
    random.shuffle(valid_coords)

    # 分配数据集（基于筛选后的有效块）
    total = len(valid_coords)
    num_train = int(split_ratio[0] * total)
    num_val = int(split_ratio[1] * total)

    splits = {
        'train': valid_coords[:num_train],
        'val': valid_coords[num_train:num_train + num_val],
        'test': valid_coords[num_train + num_val:]
    }

    # 写入记录文件
    lists = {'train': [], 'val': [], 'test': []}

    print(f'划分后 - 训练集: {len(splits["train"])}, 验证集: {len(splits["val"])}, 测试集: {len(splits["test"])}')
    for split, coords in splits.items():
        for idx, (x, y) in enumerate(tqdm(coords, desc=f'处理{split}集')):
            name = f"{split}_{idx + 1}"
            path_A = ospj(save_root, 'A', name + '.png')
            path_B = ospj(save_root, 'B', name + '.png')
            path_L = ospj(save_root, 'label', name + '.png')

            save_crop(img_A, x, y, tile_size, path_A)
            save_crop(img_B, x, y, tile_size, path_B)
            save_crop(label, x, y, tile_size, path_L)

            lists[split].append(name + '.png\n')

    # 保存list文件
    for split in ['train', 'val', 'test']:
        with open(ospj(save_root, 'list', f'{split}.txt'), 'w') as f:
            f.writelines(lists[split])

    print('✅ 裁剪、筛选与划分完成！')


if __name__ == '__main__':
    main()