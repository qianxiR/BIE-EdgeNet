import os
import cv2
import numpy as np
from tqdm import tqdm

def generate_edge_labels(label_dir, save_dir, kernel_size=3):
    """
    从变化标签生成边缘标签图像。

    Args:
        label_dir (str): 输入变化标签（二值图）的文件夹路径
        save_dir (str): 输出边缘标签图像保存路径
        kernel_size (int): 膨胀/腐蚀使用的核大小
    """
    os.makedirs(save_dir, exist_ok=True)
    label_files = [f for f in os.listdir(label_dir) if f.endswith(('.png', '.jpg', '.tif'))]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    for file in tqdm(label_files, desc="Processing edge masks"):
        label_path = os.path.join(label_dir, file)
        save_path = os.path.join(save_dir, file)

        # 读取二值标签图像
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            print(f"Failed to read {label_path}")
            continue

        # 保证是0和1或0和255的二值图
        _, binary = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)

        # 边缘提取：膨胀 - 腐蚀 = 边缘（即形态学梯度）
        dilated = cv2.dilate(binary, kernel)
        eroded = cv2.erode(binary, kernel)
        edge = cv2.absdiff(dilated, eroded)

        # 保存边缘图
        cv2.imwrite(save_path, edge)

    print("✅ Edge label generation completed.")

if __name__ == "__main__":
    # 示例路径（根据你的数据集修改）
    label_dir = r"F:\Data\LEVIR-CD\merged_data\label"   # 输入变化标签路径
    save_dir = r"F:\Data\LEVIR-CD\merged_data\label_edge"     # 输出边缘标签路径

    generate_edge_labels(label_dir, save_dir, kernel_size=2)