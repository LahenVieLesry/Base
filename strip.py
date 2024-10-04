import cv2
import math
import random
import time
import scipy
import torch
import scipy
import scipy.stats
import numpy as np
from matplotlib import pyplot as plt

from param import param
import glob
import os
from tqdm import tqdm

device = torch.device(f'cuda:{param.GPU_num}' if torch.cuda.is_available() else 'cpu')

model_path = 'models/train/000.pth'

x_train = []
train_folders = glob.glob('datasets/train/*')
for folder in train_folders:
  npy_files = glob.glob(os.path.join(folder, '*.npy'))
  for file in npy_files:
    x_train.append(np.load(file))
x_train = np.array(x_train)

x_poi_train = []
npy_files = glob.glob('datasets/temp/train_bit_layer4_predefattloss1/*.npy')
for file in npy_files:
  x_poi_train.append(np.load(file))
x_poi_train = np.array(x_poi_train)

def load_model(path):
    print('Loading a pretrained model ')
    model=torch.load(path)
    model.to(device)
    return model
model = load_model(model_path).eval()

def superimpose(background, overlay):
  added_image = cv2.addWeighted(background, 1, overlay, 1, 0)
  return (added_image)

def entropyCal(background, max_idx, n):
  entropy_sum = [0] * n
  x1_add = [0] * n
  index_overlay = np.random.randint(0, max_idx, size=n)
  for x in range(n):
    x1_add[x] = (superimpose(background, x_train[index_overlay[x]]))

  x1_add_tensor = torch.tensor(np.array(x1_add)).float().to(device)
  py1_add = model(x1_add_tensor).cpu().detach().numpy()
  
  epsilon = 1e-10  # Small value to avoid log(0)
  py1_add = np.clip(py1_add, epsilon, 1.0)  # Ensure values are within [epsilon, 1.0]
  
  EntropySum = -np.nansum(py1_add * np.log2(py1_add))
  return EntropySum

n_sample = 100
# entropy_benigh = [0] * min(2000, len(x_train))
# entropy_trojan = [0] * min(2000, len(x_poi_train))
entropy_benigh = [0] * len(x_train)
entropy_trojan = [0] * len(x_poi_train)

for j in tqdm(range(len(x_train)), desc="x_train Progress"):
  x_background = x_train[j] 
  entropy_benigh[j] = entropyCal(x_background, len(x_train)-1, n_sample)

for j in tqdm(range(len(x_poi_train)), desc="x_poi_train Progress"):
  x_poison_background = x_poi_train[j]
  entropy_trojan[j] = entropyCal(x_poison_background, len(x_poi_train)-1, n_sample)

entropy_benigh = [x / n_sample for x in entropy_benigh] # get entropy for 2000 clean inputs
entropy_trojan = [x / n_sample for x in entropy_trojan] # get entropy for 2000 trojaned inputs
print(f'entropy_benigh_shape: {len(entropy_benigh)}')
print(f'entropy_trojan_shape: {len(entropy_trojan)}')

# ============================================================================================================================
bins = 30
plt.figure(figsize=(12, 8))
plt.hist(entropy_benigh, bins, weights=np.ones(len(entropy_benigh)) / len(entropy_benigh), alpha=0.7, label='Benign', color='skyblue')
plt.hist(entropy_trojan, bins, weights=np.ones(len(entropy_trojan)) / len(entropy_trojan), alpha=0.7, label='Poison', color='salmon')
plt.legend(loc='upper right', fontsize=20)
plt.xlabel('Entropy', fontsize=20)
plt.ylabel('Probability (%)', fontsize=20)
plt.title('Normalized Entropy Distribution', fontsize=24)
plt.grid(True, linestyle='--', alpha=0.2)
plt.tick_params(labelsize=20)

fig1 = plt.gcf()
plt.show()
fig1.savefig('strip.png', bbox_inches='tight')

# ============================================================================================================================
(mu, sigma) = scipy.stats.norm.fit(entropy_benigh)
print(f'mu: {mu}, sigma: {sigma}')

threshold = scipy.stats.norm.ppf(0.01, loc = mu, scale =  sigma) #use a preset FRR of 0.01. This can be 
print(f'threshold: {threshold}')

FAR = sum(i > threshold for i in entropy_trojan)
print(f'FAR: {FAR/2000*100}%') #reproduce results in Table 3 of our paper

# ============================================================================================================================
min_benign_entropy = min(entropy_benigh)
max_trojan_entropy = max(entropy_trojan)

print(f'min_benign_entropy: {min_benign_entropy}') # check min entropy of clean inputs
print(f'max_trojan_entropy: {max_trojan_entropy}') # check max entropy of trojaned inputs