import os
import random
import shutil
import numpy as np
import torch

def same_seeds(seed: int = None) -> int:
    if seed is None:
        seed = random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)
    elif isinstance(seed, str):
        seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    return seed

same_seeds(7)

def random_delete(src_folder, rate):
    # 遍历根目录下的所有子文件夹
    for subdir, dirs, files in os.walk(src_folder):
        for dir_name in dirs:
            # 获取当前子文件夹中的所有文件路径
            dir_path = os.path.join(subdir, dir_name)
            print(dir_path)
            all_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
            
            # 计算需要删除的文件数量
            num_files_to_delete = int(len(all_files) * rate)
            
            # 随机选择文件进行删除
            files_to_delete = random.sample(all_files, num_files_to_delete)
            
            # 删除文件
            for file_path in files_to_delete:
                os.remove(file_path)

    print("Files deletion complete.")

def random_remove(src_folder, des_folder, rate):
    # 获取 src_folder 中的所有文件路径
    all_files = [os.path.join(src_folder, f) for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    
    # 计算需要移动的文件数量
    num_files_to_move = int(2086 * rate)
    # num_files_to_move = 209
    
    # 随机选择文件进行移动
    files_to_move = random.sample(all_files, num_files_to_move)
    
    # 移动文件
    for file_path in files_to_move:
        # 构建目标路径，并创建必要的目录
        backup_path = os.path.join(des_folder, os.path.basename(file_path))
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        
        # 移动文件到目标目录
        shutil.move(file_path, backup_path)
        # print(f"Moved: {file_path} -> {backup_path}")

    print("Files moved complete.")

def copy_files_to_corresponding_folders(src_folder, des_folder):
    # 遍历源根目录下的所有子文件夹
    for subdir, dirs, files in os.walk(src_folder):
        for dir_name in dirs:
            src_dir = os.path.join(subdir, dir_name)
            dest_dir = os.path.join(des_folder, dir_name)
            
            # 如果目标文件夹不存在，创建它
            os.makedirs(dest_dir, exist_ok=True)
            
            # 复制文件到目标文件夹
            for file_name in os.listdir(src_dir):
                src_file = os.path.join(src_dir, file_name)
                dest_file = os.path.join(dest_dir, file_name)
                shutil.copy2(src_file, dest_file)
                print(f"Copied: {src_file} -> {dest_file}")

    print("File copy complete.")


src_folder = 'datasets/temp/train'
des_folder = 'datasets/trigger_train'

# random_delete(src_folder, 0.6)
# random_remove(src_folder, des_folder, 0.1)
copy_files_to_corresponding_folders(src_folder, des_folder)