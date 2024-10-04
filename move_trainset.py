import os
import shutil

source_dir = os.getenv('source_dir')
target_dir = 'datasets/train_trigger/left'

print(f'Move {source_dir} --> {target_dir}')

num_remove = 0
num_copy = 0

for filename in os.listdir(source_dir):
    if filename.endswith('.npy'):
        new_filename = filename.split('_', 1)[1]
        
        src_file = os.path.join(source_dir, filename)
        dest_file = os.path.join(target_dir, filename)
        
        prefix = filename.split('_', 1)[0]
        old_file_to_remove = os.path.join('datasets/train_trigger', prefix, new_filename)
        if os.path.exists(old_file_to_remove):
            os.remove(old_file_to_remove)
            num_remove += 1
        
        shutil.copy(src_file, dest_file)
        num_copy += 1

print(f'Total removed: {num_remove}, Total copied: {num_copy}')
