import torch
import torch.nn as nn


class SimpleRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.2):
        super(SimpleRNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义SimpleRNN层
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # 定义Dropout层
        self.dropout = nn.Dropout(dropout)

        # 定义全连接层
        self.fc = nn.Linear(hidden_size, output_size)

        # 定义Softmax层
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 初始化隐藏层状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播
        out, _ = self.rnn(x, h0)

        # 仅取最后一个时间步的输出
        out = out[:, -1, :]

        # dropout层
        out = self.dropout(out)

        # 全连接层
        out = self.fc(out)

        # softmax层
        out = self.softmax(out)
        return out


