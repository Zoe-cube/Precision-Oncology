from __future__ import print_function
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score

'''
Part of format from pytorch examples repo: 
https://github.com/pytorch/examples/blob/master/mnist/main.py
'''

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.l1 = nn.Linear(3*334*216, 1024)
        #self.l2 = nn.Linear(262144, 65536)
        #self.l3 = nn.Linear(65536, 16384)
        #self.l4 = nn.Linear(16384, 1024)
        #self.l5 = nn.Linear(4096, 1024)
        self.l6 = nn.Linear(1024, 256)
        self.l7 = nn.Linear(256, 128)
        self.l8 = nn.Linear(128, 64)
        self.l9 = nn.Linear(64, 32)
        self.l10 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 3*334*216)  # 将图像张量展平成一维向量
        nn = F.relu(self.l1(x))
        #nn = F.relu(self.l2(nn))
        #nn = F.relu(self.l3(nn))
        #nn = F.relu(self.l4(nn))
        #nn = F.relu(self.l5(nn))
        nn = F.relu(self.l6(nn))
        nn = F.relu(self.l7(nn))
        nn = F.relu(self.l8(nn))
        nn = F.relu(self.l9(nn))
        nn = self.l10(nn)
        return nn

def train(model, device, train_loader, optimizer):
    model.train()
    cost = nn.CrossEntropyLoss()
    with tqdm(total=len(train_loader)) as progress_bar:
        for batch_idx, (data, label) in tqdm(enumerate(train_loader)):
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = cost(output, label)        
            loss.backward()
            optimizer.step()
            progress_bar.update(1)


def val(model, device, val_loader, checkpoint=None):
    if checkpoint is not None:
        model_state = torch.load(checkpoint)
        model.load_state_dict(model_state)     
    # Need this line for things like dropout etc.  
    model.eval()
    preds = []
    targets = []
    cost = nn.CrossEntropyLoss()
    losses = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(val_loader):
            data = data.to(device)
            target = label.clone()
            output = model(data)
            preds.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
            losses.append(cost(output, label))
    loss = np.mean(losses)
    preds = np.argmax(np.concatenate(preds), axis=1)
    targets  = np.concatenate(targets)
    acc = accuracy_score(targets, preds)
    return loss, acc

def test(model, device, test_loader, checkpoint=None):
    if checkpoint is not None:
        model_state = torch.load(checkpoint)
        model.load_state_dict(model_state)    
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data = data.to(device)
            target = label.clone()
            output = model(data)
            preds.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
        
    preds = np.argmax(np.concatenate(preds), axis=1)
    targets  = np.concatenate(targets)
    acc = accuracy_score(targets, preds)
    return acc

