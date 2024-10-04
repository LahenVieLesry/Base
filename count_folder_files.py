import os

# 设置根目录路径
root_folder = 'datasets/simple_train/yes'

# 初始化文件计数器
npy_file_count = 0

# 遍历根目录及其所有子文件夹
for root, dirs, files in os.walk(root_folder):
    for file in files:
        # 检查文件是否以 .npy 结尾
        if file.endswith('.npy'):
            npy_file_count += 1

# 输出统计结果
print(f"Total number of .npy files: {npy_file_count}")
