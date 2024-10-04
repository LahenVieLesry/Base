import os
import shutil

name = 'train'
# 指定目标文件夹路径
source_folder = f'datasets/temp/adaptive_train_suc_1_id_3'
destination_folder = f'datasets/temp/{name}/'

# 检查并创建目标文件夹
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 遍历source_folder中的所有文件
for filename in os.listdir(source_folder):
    if filename.endswith('.npy'):
        # 找到第一个 "_" 的位置
        first_underscore_index = filename.find('_')
        
        if first_underscore_index != -1:
            # 提取第一个 "_" 之前的部分作为新文件夹名
            # folder_name = filename[:first_underscore_index]
            # destination_folder = os.path.join(destination_folder, folder_name)
            
            # 创建新文件夹（如果不存在）
            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)
            
            # 去掉第一个 "_" 及其之前的部分
            new_filename = filename[first_underscore_index+1:]
            
            # 构建新文件的完整路径
            destination_file_path = os.path.join(destination_folder, new_filename)
            
            # 构建原文件的完整路径
            source_file_path = os.path.join(source_folder, filename)
            
            # 将文件复制到新的目录，并重命名
            shutil.copy2(source_file_path, destination_file_path)
            # print(f"Copied: {source_file_path} to {destination_file_path}")
