import os
from param import param


CLASSES = 'yes, no, up, down, left, right, on, off, stop, go'.split(', ')
classes = CLASSES
frame_index_list = []

folder = param.path.benign_train_wavpath
all_classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')]
#{'yes': 0, 'no': 1, 'up': 2, 'down': 3, 'left': 4, 'right': 5, 'on': 6, 'off': 7, 'stop': 8, 'go': 9}
class_to_idx = {classes[i]: i for i in range(len(classes))}
idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
print(idx_to_class[4])