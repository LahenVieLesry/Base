import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
from param import param

device = torch.device(f'cuda:{param.GPU_num}' if torch.cuda.is_available() else 'cpu')

# 定义NpyDataset类
class NpyDataset(Dataset):
    def __init__(self, folder, label, transform=None):
        self.folder = folder
        self.label = label
        self.transform = transform
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        if self.transform:
            data = self.transform(data)
        return data, self.label

# 加载模型
model_path = 'models/train/base_resnet34_posion_left_epochs_200_batchsize_128_Adam_lr_0.02_step_rate0.9_ir_iter_epoch5_mom_0.9_id_6.pth'
model = torch.load(model_path)
model.to(device)
model.eval()

# 加载训练数据集
train_folder = 'datasets/train'
train_features = []
train_labels = []
with torch.no_grad():
    for label in os.listdir(train_folder):
        folder_path = os.path.join(train_folder, label)
        if os.path.isdir(folder_path):
            dataset = NpyDataset(folder_path, label)
            data_loader = DataLoader(dataset, batch_size=128, shuffle=False)
            for data, _ in tqdm(data_loader, desc=f'Extracting {label}'):
                data = data.to(device).float()
                output = model(data)
                train_features.append(output.cpu())
                train_labels.extend([label] * data.size(0))

# 加载left数据集
# left_folder = 'datasets/temp/test_bit_c/left'
# left_folder = 'datasets/temp/train_wav_layer4'
# left_folder = 'datasets/temp/train_wav_layer4_predefattloss1'
left_features = []
left_labels = []
with torch.no_grad():
    left_dataset = NpyDataset(left_folder, 'left')
    left_data_loader = DataLoader(left_dataset, batch_size=128, shuffle=False)
    for data, _ in tqdm(left_data_loader, desc='Extracting left'):
        data = data.to(device).float()
        output = model(data)
        left_features.append(output.cpu())
        left_labels.extend(['left'] * data.size(0))

# 合并特征和标签
features = torch.cat(train_features + left_features).numpy()
labels = np.array(train_labels + left_labels)
is_poison = np.array([False for label in train_labels] + [True if label == 'left' else 0 for label in left_labels])
print(f'is_poison: {is_poison.sum()} / {len(is_poison)}')

# t-SNE 聚类
tsne = TSNE()
tsne_results = tsne.fit_transform(features)

# 绘制结果  ``
plt.figure(figsize=(10, 10))
unique_labels = np.unique(labels)
colors = plt.get_cmap('tab10', len(unique_labels))

for i, label in tqdm(enumerate(unique_labels), desc='Plotting'):
    indices = labels == label
    if label == 'left':
        poison_indices = indices & is_poison
        normal_indices = indices & ~is_poison
        plt.scatter(tsne_results[normal_indices, 0], tsne_results[normal_indices, 1], alpha=0.5, color=colors(i), s=10, label=label)
        plt.scatter(tsne_results[poison_indices, 0], tsne_results[poison_indices, 1], color='black', s=10, label=f'{label} (poison)')
    else:
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], color=colors(i), alpha=0.5, s=10, label=label)

plt.legend()
plt.title('t-SNE of Keyword Spotting Model Features')
plt.savefig(f'tsne_{os.path.basename(left_folder)}.png', dpi=300)
plt.show()