'''
This is the test code of NAD defense.
'''


from copy import deepcopy
import os.path as osp

import numpy as np
import cv2
import torch
import torch.nn as nn

from defense.NAD import NAD
from param import param
from datasets import *


# ========== Set global settings ==========
global_seed = 7
deterministic = True
torch.manual_seed(global_seed)
datasets_root_dir = '../datasets'
CUDA_VISIBLE_DEVICES = f'{param.GPU_num}'
portion = 0.05
batch_size = param.trigger_train.batch_size
num_workers = param.trigger_train.num_workers

_model_name = '000.pth'
benign_model_name = 'models/train/resnet34_epochs_200_batchsize_128_Adam_lr_0.02_mom_0.9_id_0.pth'
attack_model_name = './models/train/' + _model_name
save_model_name_benign = f'models/defense/NAD_benign_' + _model_name
save_model_name_poi = f'models/defense/NAD_poi_' + _model_name

def load_model(path):
    print('Loading a pretrained model')
    device = torch.device(f'cuda:{param.GPU_num}' if torch.cuda.is_available() else 'cpu')
    model = torch.load(path, map_location=device)
    return model


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def test_without_defense(model_name, dataset_name, attack_name, defense_name, benign_dataset, attacked_dataset, defense, y_target):
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'batch_size': batch_size,
        'num_workers': num_workers,

        'metric': 'BA',

        'save_dir': 'experiments/NAD-defense',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_BA_without_defense'
    }
    # defense.test(benign_dataset, schedule)

    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'batch_size': batch_size,
        'num_workers': num_workers,

        # 1. ASR: the attack success rate calculated on all poisoned samples
        # 2. ASR_NoTarget: the attack success rate calculated on all poisoned samples whose ground-truth labels are not the target label
        # 3. BA: the accuracy on all benign samples
        # Hint: For ASR and BA, the computation of the metric is decided by the dataset but not schedule['metric'].
        # In other words, ASR or BA does not influence the computation of the metric.
        # For ASR_NoTarget, the code will delete all the samples whose ground-truth labels are the target label and then compute the metric.
        'metric': 'ASR_NoTarget',
        'y_target': y_target,

        'save_dir': 'experiments/NAD-defense',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_ASR_without_defense'
    }
    # defense.test(attacked_dataset, schedule)


def test(model_name, dataset_name, attack_name, defense_name, benign_dataset, attacked_dataset, defense, y_target):
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'batch_size': batch_size,
        'num_workers': num_workers,

        'metric': 'BA',

        'save_dir': 'experiments/NAD-defense',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_BA'
    }
    # defense.test(benign_dataset, schedule)

    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'batch_size': batch_size,
        'num_workers': num_workers,

        # 1. ASR: the attack success rate calculated on all poisoned samples
        # 2. ASR_NoTarget: the attack success rate calculated on all poisoned samples whose ground-truth labels are not the target label
        # 3. BA: the accuracy on all benign samples
        # Hint: For ASR and BA, the computation of the metric is decided by the dataset but not schedule['metric'].
        # In other words, ASR or BA does not influence the computation of the metric.
        # For ASR_NoTarget, the code will delete all the samples whose ground-truth labels are the target label and then compute the metric.
        'metric': 'ASR_NoTarget',
        'y_target': y_target,

        'save_dir': 'experiments/NAD-defense',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_ASR'
    }
    # defense.test(attacked_dataset, schedule)

model_name, dataset_name, attack_name, defense_name = 'resnet34', 'dataset_name', 'Benign', 'NAD'
model = load_model(benign_model_name)

transform_train = None
trainset = SpeechCommandsDataset(param.path.benign_train_npypath)
transform_test = None
testset = SpeechCommandsDataset(param.path.benign_test_npypath)

defense = NAD(
    model=model,
    loss=nn.CrossEntropyLoss(),
    power=2.0, 
    beta=[500, 500, 500],
    target_layers=['layer2', 'layer3', 'layer4'],
    seed=global_seed,
    deterministic=deterministic
)
test_without_defense(model_name, dataset_name, attack_name, defense_name, testset, testset, defense, None)

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    'batch_size': batch_size,
    'num_workers': num_workers,

    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,

    'tune_lr': 0.01,
    'tune_epochs': 10,

    'epochs': 20,
    'schedule':  [2, 4, 6, 8], 

    'log_iteration_interval': 20,
    'test_epoch_interval': 20,
    'save_epoch_interval': 20,

    'save_dir': 'experiments/NAD-defense',
    'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_p{portion}'
}

defense.repair(dataset=trainset, portion=portion, schedule=schedule)
repaired_model = defense.get_model()
torch.save(repaired_model, save_model_name_benign)
test(model_name, dataset_name, attack_name, defense_name, testset, testset, defense, None)


# ===================== BadNets ======================
attack_name = 'wav'
poi_model = load_model(attack_model_name)

poisoned_trainset = SpeechCommandsDataset(param.path.poison_train_path)
poisoned_testset = SpeechCommandsDataset('datasets/temp/test_bit_c/')

defense = NAD(
    model=model,
    loss=nn.CrossEntropyLoss(),
    power=2.0, 
    beta=[500, 500, 500],
    target_layers=['layer2', 'layer3', 'layer4'],
    seed=global_seed,
    deterministic=deterministic
)
test_without_defense(model_name, dataset_name, attack_name, defense_name, testset, poisoned_testset, defense, 1)

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    'batch_size': batch_size,
    'num_workers': num_workers,

    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,

    'tune_lr': 0.01,
    'tune_epochs': 10,

    'epochs': 20,
    'schedule':  [2, 4, 6, 8], 

    'log_iteration_interval': 20,
    'test_epoch_interval': 20,
    'save_epoch_interval': 20,

    'save_dir': 'experiments/NAD-defense',
    'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_p{portion}'
}

defense.repair(dataset=trainset, portion=portion, schedule=schedule)
repaired_model = defense.get_model()
torch.save(repaired_model, save_model_name_poi)
test(model_name, dataset_name, attack_name, defense_name, testset, poisoned_testset, defense, 1)

