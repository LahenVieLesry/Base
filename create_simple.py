import os
import shutil
import random

import numpy as np
import torch

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
       torch.cuda.manual_seed(seed)
       torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    np.random.seed(seed) # Numpy module.
    random.seed(seed) # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seeds(7)
    
def process(source_dir, target_dir):
    # 创建目标文件夹，如果不存在则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历源文件夹中的每个子文件夹
    for subdir in os.listdir(source_dir):
        source_subdir = os.path.join(source_dir, subdir)
        
        # 检查子文件夹是否为目录
        if os.path.isdir(source_subdir):
            target_subdir = os.path.join(target_dir, subdir)
            
            # 如果目标子文件夹不存在，创建它
            if not os.path.exists(target_subdir):
                os.makedirs(target_subdir)

            # 获取子文件夹中的所有文件
            files = os.listdir(source_subdir)
            
            # 随机选择40%的文件
            num_files_to_copy = int(len(files) * 0.4)
            selected_files = random.sample(files, num_files_to_copy)

            # 复制选中的文件到目标目录
            for file_name in selected_files:
                source_file = os.path.join(source_subdir, file_name)
                target_file = os.path.join(target_subdir, file_name)
                
                # 复制文件
                shutil.copy2(source_file, target_file)

            print(f"Copied {num_files_to_copy} files from {subdir} to {target_subdir}.")

process('./datasets/train', './datasets/simple_train')
process('./datasets/test', './datasets/simple_test')
