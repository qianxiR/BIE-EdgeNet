import os
import os.path as osp
import shutil
import sys
import numpy as np
import cv2
import time
from tqdm import tqdm

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def _get_paths_from_images(path):
    assert osp.isdir(path), f'{path} is not a valid directory'
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, f'{path} has no valid image file'
    return images

def rager(p, img_size=256, size=4):
    img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    a = img[0:size, :]
    b = img[img_size - size:img_size, :]
    c = img[:, 0:size]
    d = img[:, img_size - size:img_size]
    e = cv2.countNonZero(a) + cv2.countNonZero(b) + cv2.countNonZero(c) + cv2.countNonZero(d)
    rate2 = (size * img_size * 4 - e) * 100 / (img_size * img_size)
    return rate2

def extract_signle(opt):
    GT_folder = opt['input_folder']
    save_GT_folder = opt['save_folder']
    label_dir = osp.join(GT_folder, 'label')
    img_list = _get_paths_from_images(label_dir)
    out_label_dir = osp.join(save_GT_folder, 'label')
    out_before_dir = osp.join(save_GT_folder, 'A')
    out_after_dir = osp.join(save_GT_folder, 'B')
    os.makedirs(out_label_dir, exist_ok=True)
    os.makedirs(out_before_dir, exist_ok=True)
    os.makedirs(out_after_dir, exist_ok=True)

    all_patches = {'train': [], 'val': [], 'test': []}

    for path in tqdm(img_list, desc="裁剪中"):
        img_name = osp.basename(path)
        prefix = img_name.split('_')[0]  # train_1 → train
        base_dir = osp.dirname(osp.dirname(path))

        img_A_path = osp.join(base_dir, 'A', img_name)
        img_B_path = osp.join(base_dir, 'B', img_name)

        img_A = cv2.imread(img_A_path, cv2.IMREAD_UNCHANGED)
        img_B = cv2.imread(img_B_path, cv2.IMREAD_UNCHANGED)
        img_L = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if img_A is None or img_B is None or img_L is None:
            print(f"[跳过] 图像不存在: {img_name}")
            continue

        h, w = img_L.shape[:2]
        crop_sz = opt['crop_sz']
        step = opt['step']
        thres_sz = opt['thres_sz']
        classtype = opt['classtype']

        h_space = np.arange(0, h - crop_sz + 1, step)
        if h - (h_space[-1] + crop_sz) > thres_sz:
            h_space = np.append(h_space, h - crop_sz)
        w_space = np.arange(0, w - crop_sz + 1, step)
        if w - (w_space[-1] + crop_sz) > thres_sz:
            w_space = np.append(w_space, w - crop_sz)

        index = 0
        for x in h_space:
            for y in w_space:
                crop_img_L = img_L[x:x + crop_sz, y:y + crop_sz]
                crop_img_A = img_A[x:x + crop_sz, y:y + crop_sz]
                crop_img_B = img_B[x:x + crop_sz, y:y + crop_sz]

                crop_img_L = np.ascontiguousarray(crop_img_L)
                crop_img_A = np.ascontiguousarray(crop_img_A)
                crop_img_B = np.ascontiguousarray(crop_img_B)

                index += 1
                name = osp.splitext(img_name)[0] + f"_s{index:03d}.png"

                if classtype is not None:
                    crop_img_L[crop_img_L == classtype] = 255
                    crop_img_L[crop_img_L != 255] = 0

                cv2.imwrite(osp.join(out_before_dir, name), crop_img_A, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                cv2.imwrite(osp.join(out_after_dir, name), crop_img_B, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                cv2.imwrite(osp.join(out_label_dir, name), crop_img_L, [cv2.IMWRITE_PNG_COMPRESSION, 3])

                all_patches[prefix].append(name)
    return all_patches

def filter_patches(all_patches, base_path):
    result = {'train': [], 'val': [], 'test': []}
    for prefix in ['train', 'val', 'test']:
        for name in tqdm(all_patches[prefix], desc=f"筛查 {prefix}"):
            if prefix == 'test':
                result['test'].append(name)
                continue
            file1 = osp.join(base_path, 'label', name)
            file2 = osp.join(base_path, 'A', name)
            file3 = osp.join(base_path, 'B', name)
            img = cv2.imread(file1, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            a = img[0, :]
            b = img[255, :]
            c = img[:, 0]
            d = img[:, 255]
            e = cv2.countNonZero(a + b + c + d)
            f = (cv2.countNonZero(img) * 100) / (256 * 256)
            if f < 0.2 or (e > 10 and f < 0.9) or rager(file3) > 0.5:
                os.remove(file1)
                os.remove(file2)
                os.remove(file3)
            else:
                result[prefix].append(name)
    return result

def write_lists(filtered_dict, base_path):
    list_path = osp.join(base_path, 'list')
    os.makedirs(list_path, exist_ok=True)
    for split in ['train', 'val', 'test']:
        with open(osp.join(list_path, f"{split}.txt"), 'w') as f:
            for name in filtered_dict[split]:
                f.write(name + '\n')

def main(GT_folder, save_GT_folder, img_size=256, classtype=None):
    opt = {
        'n_thread': 20,
        'crop_sz': img_size,
        'step': img_size,
        'thres_sz': img_size,
        'input_folder': GT_folder,
        'save_folder': save_GT_folder,
        'classtype': classtype
    }

    print('开始裁剪...')
    all_patches = extract_signle(opt)
    print('裁剪完成，开始筛查...')
    filtered = filter_patches(all_patches, save_GT_folder)
    print('写入 list 文件...')
    write_lists(filtered, save_GT_folder)
    print('全部完成')

if __name__ == '__main__':
    split_in_list = False
    img_size = 256
    crop_path = r'E:\zyh\Data\Building change detection dataset\WHU_1024'
    out_path = r'E:\zyh\Data\Building change detection dataset\WHU_256'
    time_start = time.time()
    classtype = 255

    main(crop_path, out_path, img_size=img_size, classtype=classtype)

    time_end = time.time()
    print('用时', round(time_end - time_start, 1), '秒')
