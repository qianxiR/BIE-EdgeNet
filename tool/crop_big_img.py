import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from os.path import join as ospj

# ====== 配置参数 ======
tile_size = 256
stride = 256
split_ratio = [0.7, 0.2, 0.1]
random_seed = 42

# 输入路径
img_A_path = r"E:\CDCD\zyhCD20160607\Level18\zyhCD20160607.tif"
img_B_path = r"E:\CDCD\zyhCD20230308\Level18\zyhCD20230308.tif"
label_path = r"E:\CDCD\CDCDLabel.tif"

# 输出路径
save_root = r'E:\zyh\Data\CDCD\CDCD_256_7_1_2'
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

def save_crop(img, x, y, size, path):
    crop = img[y:y+size, x:x+size]
    cv2.imwrite(path, crop)

# ====== 主处理流程 ======
def main():
    # 读取图像
    img_A = cv2.imread(img_A_path)
    img_B = cv2.imread(img_B_path)
    label = cv2.imread(label_path, 0)  # 单通道读取

    assert img_A.shape[:2] == img_B.shape[:2] == label.shape[:2], "尺寸不一致"

    coords = slide_crop(img_A, size=tile_size, stride=stride)
    random.seed(random_seed)
    random.shuffle(coords)

    # 分配数据集
    total = len(coords)
    num_train = int(split_ratio[0] * total)
    num_val = int(split_ratio[1] * total)

    splits = {
        'train': coords[:num_train],
        'val': coords[num_train:num_train+num_val],
        'test': coords[num_train+num_val:]
    }

    # 写入记录文件
    lists = {'train': [], 'val': [], 'test': []}

    print(f'总共裁剪图块数量: {total}')
    for split, coords in splits.items():
        for idx, (x, y) in enumerate(tqdm(coords, desc=f'{split}')):
            name = f"{split}_{idx+1}"
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

    print('✅ 裁剪与划分完成！')

if __name__ == '__main__':
    main()
