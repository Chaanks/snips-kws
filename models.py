import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, 8, bidirectional=False)
        self.fc1 = nn.Linear(8 * 10, 80)
        self.fc2 = nn.Linear(80, output_size)

    def forward(self, x):
        #print(x.shape)
        #print(x.view(x.shape[0], x.shape[1], -1).shape)
        lstm_out, _ = self.lstm(x.view(x.shape[0], x.shape[1], -1))

        x = self.fc1(lstm_out.view(lstm_out.shape[0], -1))
        #x = F.relu(self.fc1(lstm_out[:, 6]))
        x = self.fc2(x)

        #prob = F.log_softmax(x, dim=1)
        return x