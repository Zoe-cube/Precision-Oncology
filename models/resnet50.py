from __future__ import print_function
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from tqdm import tqdm
from sklearn.metrics import accuracy_score,roc_auc_score, f1_score
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)# batch normalization no need this
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = torch.add(out, identity)
        out = self.relu(out)

        return out





class Bottleneck(nn.Module):
    # The multiple of the extension in each stage dimension
    extention=4
    def __init__(self,inplanes,planes,stride,downsample=None):
        '''

        :param inplanes:  the number of channels before the block
        :param planes: The number of channels when processing in the middle of the block
                planes*self.extention:output features
        :param stride:
        :param downsample:
        '''
        super(Bottleneck, self).__init__()

        self.conv1=nn.Conv2d(inplanes,planes,kernel_size=1,stride=stride,bias=False)
        self.bn1=nn.BatchNorm2d(planes)

        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(planes)

        self.conv3=nn.Conv2d(planes,planes*self.extention,kernel_size=1,stride=1,bias=False)
        self.bn3=nn.BatchNorm2d(planes*self.extention)

        # self.relu=nn.ReLU(inplace=True)

        #Determine whether the residuals have convolution
        self.downsample=downsample
        self.stride=stride

    def forward(self,x):

        residual=x

        #convolution
        out=self.conv1(x)
        out=self.bn1(out)
        out = F.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)
        out = F.relu(out)

        out=self.conv3(out)
        out=self.bn3(out)
        out = F.relu(out)

        #Whether to be directly connected
        # (if Indentity blobk is directly connected;
        # If the Conv2 Block needs to convolve the residual edge,
        # change the number of channels and the size
        if self.downsample is not None:
            residual=self.downsample(x)

        #Add the residual part and the convolution part
        #out+=residual
        out=torch.add(out,residual)
        out = F.relu(out)

        return out


class net(nn.Module):
    def __init__(self,block,layers,num_class=4):
        #inplane=current channels of fm
        self.inplane=64
        super(net, self).__init__()

        #parameters
        self.block=block
        self.layers=layers

        #stem networks
        self.conv1=nn.Conv2d(3,self.inplane,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(self.inplane)
        # self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        #64,128,256,512Refers to the dimension before the 4-fold expansion, that is,
        # the dimension in the middle of the Identity Block
        self.stage1=self.make_layer(self.block,64,layers[0],stride=1)
        self.stage2=self.make_layer(self.block,128,layers[1],stride=2)
        self.stage3=self.make_layer(self.block,256,layers[2],stride=2)
        self.stage4=self.make_layer(self.block,512,layers[3],stride=2)

        #Subsequent network
        self.avgpool=nn.AvgPool2d(7)
        self.fc=nn.Linear(512*block.extention,num_class)

    def forward(self,x):
        #stem: conv+bn+maxpool
        out=self.conv1(x)
        out=self.bn1(out)
        out = F.relu(out)
        out=self.maxpool(out)

        #block
        out=self.stage1(out)
        out=self.stage2(out)
        out=self.stage3(out)
        out=self.stage4(out)

        #classify
        out=self.avgpool(out)
        out=torch.flatten(out,1)
        out=self.fc(out)

        return out

    def make_layer(self,block,plane,block_num,stride=1):
        '''
        :param block: block
        :param plane: The dimension of the intermediate operation in each module is generally equal to the output dimension /4
        :param block_num: times of repetition
        :param stride: stride
        :return:
        '''
        block_list=[]
        #Calculate whether to add downsample
        downsample=None
        if(stride!=1 or self.inplane!=plane*block.extention):
            downsample=nn.Sequential(
                nn.Conv2d(self.inplane,plane*block.extention,stride=stride,kernel_size=1,bias=False),
                nn.BatchNorm2d(plane*block.extention)
            )

        # Conv BlockThe input and output dimensions (number of channels and size) of Conv Block are not the same, so they cannot be connected continuously in series. Its role is to change the dimension of the network
        # Identity Block has the same input dimension and output (number of channels and size) and can be directly connected in series to deepen the network. # Identity Block has the same input dimension and output (number of channels and size) and can be directly connected in series to deepen the network
        #Conv_block
        conv_block=block(self.inplane,plane,stride=stride,downsample=downsample)
        block_list.append(conv_block)
        self.inplane=plane*block.extention

        #Identity Block
        for i in range(1,block_num):
            block_list.append(block(self.inplane,plane,stride=1))

        return nn.Sequential(*block_list)


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
    targets = np.concatenate(targets)
    acc = accuracy_score(targets, preds)
    auc_score = roc_auc_score(targets, preds)
    f_score = f1_score(targets, (preds > 0.5).astype(int))
    # 计算 ROC 曲线的真阳率和假阳率
    fpr, tpr, thresholds = roc_curve(targets, preds)
    return (acc, auc_score,f_score,fpr, tpr, thresholds)






# resnet=net(Bottleneck,[3,4,6,3],10)
# x=torch.randn(64,3,224,224)
# X=resnet(x)
# print(X.shape)

