from __future__ import print_function
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

import os
import argparse
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import torch
import torch.nn as nn


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.lstm1 = nn.LSTM(845, 100, batch_first=True)
        self.lstm2 = nn.LSTM(100, 500, batch_first=True)
        self.lstm3 = nn.LSTM(500, 1000, batch_first=True)
        self.leakyrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.2)
        self.batchnorm = nn.BatchNorm1d(
            1000)  # 注意：BatchNormalization 的输入维度应该是 2D 张量，这里假设输入是一个批次的数据，因此使用 nn.BatchNorm1d
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1000, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):

        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out = self.leakyrelu(out)
        out = self.dropout(out)
        out = self.batchnorm(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.leakyrelu(out)
        out = self.fc2(out)
        return out




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
            label = label.to(device)
            target = label.clone()
            output = model(data)
            preds.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
            losses.append(cost(output, label).cpu().numpy())
    loss = np.mean(losses)
    preds = np.argmax(np.concatenate(preds), axis=1)
    targets = np.concatenate(targets)
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
    targets = np.concatenate(targets)
    acc = accuracy_score(targets, preds)
    return acc
