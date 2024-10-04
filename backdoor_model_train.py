import os
import subprocess

from tqdm import tqdm
from param import param as param
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import argparse
import time

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

import torchvision
import librosa
from torchvision.transforms import *

from tensorboardX import SummaryWriter

import models
from datasets import *
import random
from log import log

import numpy as np


c = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
#-----------parameters------------

batch_size = param.trigger_train.batch_size
num_workers = param.trigger_train.num_workers
epochs = param.trigger_train.epochs
lr = param.trigger_train.lr
momentum = param.trigger_train.momentum
optim = param.trigger_train.optim
patience=param.trigger_train.patience
verbose=param.trigger_train.verbose
step_rate=param.trigger_train.step_rate
ir_iter_epoch=param.trigger_train.ir_iter_epoch

start_early_stopping = param.train.start_early_stopping
resume = param.trigger_train.resume
resume_model_name = param.trigger_train.resume_model_name
model_name = param.trigger_train.model_name
model_save_name = f"./models/train/{model_name}_posion_{param.trigger_gen.target_label}_epochs_{epochs}_batchsize_{batch_size}_{optim}_lr_{lr}_step_rate{step_rate}_ir_iter_epoch{ir_iter_epoch}_mom_{momentum}_id_{param.trigger_train.id}.pth"

device = torch.device(f'cuda:{param.GPU_num}' if torch.cuda.is_available() else 'cpu')

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

def load_model(path):
    print('Loading a pretrained model ')
    model=torch.load(path)
    return model

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}) ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

if(resume == True):
    model = load_model(resume_model_name)
else:
    model = models.create_model(model_name=model_name, num_classes=10, in_channels=1).to(device)

use_gpu = torch.cuda.is_available()
print('use_gpu', use_gpu, device)


train_dataset = SpeechCommandsDataset(param.path.poison_train_path)
test_dataset =  SpeechCommandsDataset(param.path.benign_test_npypath)

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)


criterion = torch.nn.CrossEntropyLoss()
if param.train.optim == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
elif param.train.optim == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
elif param.train.optim == 'RMSprop':
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

# Add a learning rate scheduler
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: step_rate ** (epoch // ir_iter_epoch))

early_stopping = EarlyStopping(patience=patience, verbose=verbose)

loss_list = []
accuracy_list = []

def train(epochs):
    with tqdm(range(epochs)) as _tqdm:
        for epoch in _tqdm:
            model.train()
            epoch_loss = 0
            for batch_index, (inputs, target) in enumerate(train_loader):
                print(target)
                inputs, target = inputs.to(device), target.to(device)
                inputs = inputs.type(torch.FloatTensor).to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, target)
                epoch_loss = epoch_loss + loss.item()
                loss.backward()
                optimizer.step()
                _tqdm.set_postfix(batch_index=batch_index+1, loss=loss.item(), epoch_loss=epoch_loss)
            scheduler.step()  # Update the learning rate every epoch
            if epoch >= start_early_stopping:
                val_loss = validate(model, criterion, test_loader, device)
                # early_stopping(val_loss, model)
                # if early_stopping.early_stop:
                #     print("Early stopping")
                #     log(f'trigger{param.trigger_gen.folder_name}', f'Epoch: {epoch}')
                #     f = open('trigger_train.txt','a')
                #     f.write(f'Epoch: {epoch}')
                #     break

def validate(model, criterion, dataloader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.type(torch.FloatTensor).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(dataloader)

def test(model):
    correct = 0
    total = 0
    model = load_model(model)
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.type(torch.FloatTensor).to(device)
            outputs = model(inputs)
            _, predicts = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicts == labels).sum().item()
    # print('Model name is:' + model_save_name)
    # print(f'Accuracy on test sets: {correct} / {total} = ' + '%.3f%%' % (100 * correct / total))
    # print('trigger_pattern:', str(param.trigger_gen.trigger_pattern))
    print('-----------------------------------------------------------')
    f = open('trigger_train.txt','a')
    pn = int(subprocess.check_output(f"find datasets/temp/train{param.trigger_gen.folder_name} -type f | wc -l", shell=True).decode('utf-8').strip())
    _t = int(subprocess.check_output(f"find datasets/speech_commands/train/ -type f | wc -l", shell=True).decode('utf-8').strip())
    f.write('\n\n' + '*' * 100 + f'\nfolder_name: {param.trigger_gen.folder_name}\n' + f'poison_test_path: {param.path.poison_test_path}\n' + f'model_save_name: {model_save_name}\n' + f'poison sample number: {param.trigger_gen.max_sample}\n' + f'trigger_pattern: {param.trigger_gen.trigger_pattern}\n' + f'ACC: {correct} / {total} = '+ '%.3f%%\n' % (100 * correct / total))
    log(f'trigger{param.trigger_gen.folder_name}', f'\nmodel_save_name: {model_save_name}\n' + f'folder_name: {param.trigger_gen.folder_name}\n' + f'poison_test_path: {param.path.poison_test_path}\n' + f'trigger_pattern: {param.trigger_gen.trigger_pattern}\n' + f'Poison Rate: {pn} / {_t} = ' + '%.3f%%\n' % (100 * pn / _t) + f'ACC: {correct} / {total} = '+ '%.3f%%\n' % (100 * correct / total))
    accuracy_list.append(correct / total)

def test_single(audio,model):
    model = load_model(model)
    model.eval()
    inputs = np.load(audio)
    inputs = torch.tensor(inputs)
    inputs = torch.unsqueeze(inputs, 0)
    inputs = inputs.to(device)
    result = model(inputs)
    print(c[result.argmax().item()])


if __name__ == '__main__':
    same_seeds(7)
    train(epochs)
    torch.save(model, model_save_name)
    test(model_save_name)
