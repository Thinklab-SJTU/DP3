import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np


class RMTPP(nn.Module):
    def __init__(self, cfg, args):
        super(RMTPP, self).__init__()
        self.cfg = cfg
        self.args = args
        if self.cfg.EMB_DIM != 0:
            self.embedding = nn.Embedding(self.cfg.EVENT_CLASSES, self.cfg.EMB_DIM)
            self.dropout = nn.Dropout(self.cfg.EMB_DROPOUT)
            self.lstm = nn.LSTM(self.cfg.EMB_DIM + 1, self.cfg.RNN_HIDDEN_DIM, self.cfg.RNN_LAYERS)
        else:
            self.lstm = nn.LSTM(self.cfg.EVENT_CLASSES + 1, self.cfg.RNN_HIDDEN_DIM, self.cfg.RNN_LAYERS)
        self.mlp = nn.Linear(self.cfg.RNN_HIDDEN_DIM, self.cfg.MLP_DIM)
        self.event_linear = nn.Linear(self.cfg.MLP_DIM, self.cfg.EVENT_CLASSES)
        self.time_linear = nn.Linear(self.cfg.MLP_DIM, 1)

    def forward(self, input, length):
        time_sequences = torch.tensor(input[:, :, 0:1], dtype=torch.float, device=self.args.device)
        event_sequences = torch.tensor(input[:, :, 1:], dtype=torch.long, device=self.args.device)
        if self.cfg.EMB_DIM != 0:
            event_embedding = self.embedding(event_sequences)
            event_embedding_dropout = self.dropout(event_embedding)
            time_event_input = torch.cat((time_sequences, event_embedding_dropout), 2)
        else:
            event_one_hot = torch.zeros(event_sequences.shape[0], event_sequences.shape[1], self.cfg.EVENT_CLASSES,
                                        dtype=torch.float, device=self.args.device).scatter_(2, event_sequences, 1.0)
            time_event_input = torch.cat((time_sequences, event_one_hot), 2)
        time_event_input_packed = nn.utils.rnn.pack_padded_sequence(time_event_input, length, batch_first=True)
        h0 = torch.zeros(self.cfg.RNN_LAYERS, time_event_input.shape[0], self.cfg.RNN_HIDDEN_DIM,
                         dtype=torch.float, device=self.args.device, requires_grad=True)
        c0 = torch.zeros(self.cfg.RNN_LAYERS, time_event_input.shape[0], self.cfg.RNN_HIDDEN_DIM,
                         dtype=torch.float, device=self.args.device, requires_grad=True)
        output_packed, hidden = self.lstm(time_event_input_packed, (h0, c0))
        output, _ = nn.utils.rnn.pad_packed_sequence(output_packed, batch_first=True)
        output_mlp = torch.sigmoid(self.mlp(output[:, -1, :]))
        event_output = self.event_linear(output_mlp)
        event_output = F.log_softmax(event_output, dim=1)
        time_output = self.time_linear(output_mlp)
        return time_output, event_output


class RMTPPLoss(nn.Module):
    def __init__(self, cfg, args):
        super(RMTPPLoss, self).__init__()
        self.cfg = cfg
        self.args = args
        self.intensity_w = nn.Parameter(torch.tensor(0.1, dtype=torch.float, device=args.device))
        self.intensity_b = nn.Parameter(torch.tensor(0.1, dtype=torch.float, device=args.device))
        event2index = pickle.load(open(self.cfg.EVENT_INDEX_FILE, 'rb'))
        event_stat = np.zeros(len(event2index), dtype=np.float32)
        for event in event2index.values():
            event_stat[event['index']] = event['cnt']
        event_stat = event_stat.sum() / event_stat
        event_stat = torch.from_numpy(event_stat)
        event_stat = event_stat.to(args.device)
        self.event_loss = nn.NLLLoss(weight=event_stat)

    def forward(self, output, target):
        time_target = torch.tensor(target[:, 0], dtype=torch.float, device=self.args.device)
        event_target = torch.tensor(target[:, 1], dtype=torch.long, device=self.args.device)
        time_output = output[0].squeeze()
        event_output = output[1]
        time = -1 * torch.mean(time_output + self.intensity_w * time_target + self.intensity_b +
                          (torch.exp(time_output + self.intensity_b) -
                           torch.exp(time_output + self.intensity_w * time_target + self.intensity_b)) / self.intensity_w)
        event = self.event_loss(event_output, event_target)
        return time, event, self.cfg.LOSS_ALPHA * time + event
