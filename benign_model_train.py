import os
import random

from tqdm import tqdm
from param import param as param
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import time
import numpy as np
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



c = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
#-----------parameters------------

batch_size = param.train.batch_size
num_workers = param.train.num_workers
epochs = param.train.epochs
lr = param.train.lr
momentum = param.train.momentum
optim = param.train.optim
patience=param.train.patience
verbose=param.train.verbose
step_rate=param.train.step_rate
ir_iter_epoch=param.train.ir_iter_epoch

start_early_stopping = param.train.start_early_stopping
resume = param.train.resume
resume_model_name= param.train.resume_model_name
model_name= param.train.model_name
model_save_name = './models/train/' + model_name + '_' + 'epochs_' + str(epochs) + '_' +'batchsize_' + str(batch_size) + '_' + optim + '_' + 'lr_' + str(lr)+ '_' + 'mom_' + str(momentum) + '_'  + 'id_' + param.train.id + '.pth'

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
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

if(resume == True):
    model = load_model(resume_model_name)
else:
    model = models.create_model(model_name=model_name, num_classes=10, in_channels=1).to(device)

use_gpu = torch.cuda.is_available()
print('use_gpu', use_gpu, device)
   
train_dataset = SpeechCommandsDataset(param.path.benign_train_npypath)
test_dataset = SpeechCommandsDataset(param.path.benign_test_npypath)

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)

criterion = torch.nn.CrossEntropyLoss()
if param.train.optim == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
elif param.train.optim == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: step_rate ** (epoch // ir_iter_epoch))

early_stopping = EarlyStopping(patience=patience, verbose=verbose)

loss_list = []
accuracy_list = []

def train(epochs):
    model.train()
    with tqdm(range(epochs)) as _tqdm:
        for epoch in _tqdm:
            epoch_loss = 0
            for batch_index, (inputs, target) in enumerate(train_loader):
                inputs, target = inputs.to(device), target.to(device)
                inputs = inputs.type(torch.FloatTensor).to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, target)
                epoch_loss = epoch_loss + loss.item()
                loss.backward()
                optimizer.step()
                _tqdm.set_postfix(batch_index=batch_index+1, loss=loss.item(), epoch_loss=epoch_loss)
            scheduler.step()
            if epoch >= start_early_stopping:
                val_loss = validate(model, criterion, test_loader, device)
                # early_stopping(val_loss, model)
                # if early_stopping.early_stop:
                #     print("Early stopping")
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
        for data in tqdm(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.type(torch.FloatTensor).to(device)
            outputs = model(inputs)
            _, predicts = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicts == labels).sum().item()
    print(model_save_name)
    print('Accuracy on test sets: %.3f%%' % (100 * correct / total))
    print('Total/Correct: [', total, '/', correct, ']')
    f = open('train.txt','a')
    f.write(model_save_name + '------' + '%.3f%% ' % (100 * correct / total) + '\n')
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
