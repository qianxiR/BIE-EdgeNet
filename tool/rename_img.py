import os

# -------------------------- 必改：你的文件所在文件夹路径 --------------------------
file_dir = r"E:\zyh\Data\CDCD\CDCD_test\CDCDrepair-CDCD-9-1-10\pred_result"  # 替换为你实际的文件目录（Windows路径加r''）
# -----------------------------------------------------------------------------------

# 步骤1：先预览要重命名的文件（确认无误再执行重命名）
print("=== 待重命名文件预览 ===")
rename_list = []
for filename in os.listdir(file_dir):
    # 筛选以_binary_label.png结尾的文件
    if filename.endswith("_binary_label.png"):
        # 提取前缀（如test_1、train_1）
        prefix = filename.split(".png_binary_label.png")[0]
        new_filename = f"{prefix}.png"
        rename_list.append((filename, new_filename))
        print(f"原文件：{filename} → 新文件：{new_filename}")

# 步骤2：执行批量重命名（预览无误后取消下面的注释）
print("\n=== 开始重命名 ===")
for old_name, new_name in rename_list:
    old_path = os.path.join(file_dir, old_name)
    new_path = os.path.join(file_dir, new_name)
    os.rename(old_path, new_path)
    print(f"已重命名：{old_name} → {new_name}")
print("=== 重命名完成 ===")