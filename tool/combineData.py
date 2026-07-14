import os
import shutil

# 源数据集的三个子文件夹
splits = ['train', 'val', 'test']
base_dir = r'E:\zyh\Data\LEVIR-CD'  # 数据集根目录（可修改为你的路径）
merged_dir = r'E:\zyh\Data\LEVIR-CD\512merged_data'

# 创建目标结构
for subfolder in ['A', 'B', 'label', 'list']:
    os.makedirs(os.path.join(merged_dir, subfolder), exist_ok=True)

for split in splits:
    a_src = os.path.join(base_dir, split, 'A')
    b_src = os.path.join(base_dir, split, 'B')
    label_src = os.path.join(base_dir, split, 'label')

    a_dst = os.path.join(merged_dir, 'A')
    b_dst = os.path.join(merged_dir, 'B')
    label_dst = os.path.join(merged_dir, 'label')

    list_file_path = os.path.join(merged_dir, 'list', f'{split}.txt')
    with open(list_file_path, 'w') as list_file:
        for filename in sorted(os.listdir(a_src)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                # 写入 list 文件
                list_file.write(f"{filename}\n")

                # 复制 A、B、label 图像
                shutil.copy(os.path.join(a_src, filename), os.path.join(a_dst, filename))
                shutil.copy(os.path.join(b_src, filename), os.path.join(b_dst, filename))
                shutil.copy(os.path.join(label_src, filename), os.path.join(label_dst, filename))

print("✅ 数据合并完成，list 文件生成完毕。")