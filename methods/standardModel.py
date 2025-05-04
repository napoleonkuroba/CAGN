import torch.nn as nn
from .gnn import GraphConvolution
import torch
from .gnn import GATNetwork
from methods.AutoCorrelation import AutoCorrelationLayer,AutoCorrelation


class GNNEncoder(nn.Module):
    def __init__(self, nodes, nfeat, nhead):
        super(GNNEncoder, self).__init__()
        self.gcn = GraphConvolution(nfeat, nfeat)
        self.gat = GATNetwork(nfeat, nfeat, nhead)
        self.atten = AutoCorrelationLayer(AutoCorrelation(True, 1, attention_dropout=0.05),nfeat, 1)
        self.fc = nn.Linear(nfeat*nodes, nfeat)
        self.lstm = nn.LSTM(nfeat, nfeat, 2, batch_first=True)
        self.nodes = nodes
        self.nfeat = nfeat

    def forward(self, x, adj):
        output = torch.empty(0)

        # Model
        # input,_ = self.atten(x, x, x)
        # for i in range(len(x)):
        #     graphX = input[i].reshape(-1, self.nfeat).float()
        #     # 图特征分析
        #     graphAdj = adj[i].reshape(-1, self.nodes).float()
        #     midX = self.gcn(graphX, graphAdj)
        #     output = torch.cat([output, midX], dim=0)
        #
        # useful_input = x[:, 0, :]
        # output = self.fc(output.reshape(-1, self.nfeat * self.nodes))
        # return output, useful_input, output

        # AC only
        # input,_ = self.atten(x, x, x)
        # useful_input = x[:, 0, :]
        # output = self.fc(input.reshape(-1, self.nfeat * self.nodes))
        # return output, useful_input, output

        # GNN only
        for i in range(len(x)):
            graphX = x[i].reshape(-1, self.nfeat).float()
            # 图特征分析
            graphAdj = adj[i].reshape(-1, self.nodes).float()
            midX = self.gcn(graphX, graphAdj)
            output = torch.cat([output, midX], dim=0)

        useful_input = x[:, 0, :]
        output = self.fc(output.reshape(-1, self.nfeat * self.nodes))
        return output, useful_input, output


class GNNDecoder(nn.Module):
    def __init__(self, nodes, nfeat, nhead):
        super(GNNDecoder, self).__init__()
        self.gcn = GraphConvolution(nfeat, nfeat)
        self.gat = GATNetwork(nfeat, nfeat, nhead)
        self.atten = AutoCorrelationLayer(AutoCorrelation(True, 1, attention_dropout=0.05), nfeat, 1)
        self.fc = nn.Linear(nfeat, nfeat*nodes)
        self.lstm = nn.LSTM(nfeat, nfeat, 2, batch_first=True)
        self.nodes = nodes
        self.nfeat = nfeat

    def forward(self, x):
        output = self.fc(x).reshape(-1,self.nodes,self.nfeat)
        output,_ = self.atten(output, output, output)
        return output


class GNNModel(nn.Module):
    def __init__(self, nodes, nfeat, nhead):
        super(GNNModel, self).__init__()
        self.encoder = GNNEncoder(nodes=nodes, nfeat=nfeat, nhead=nhead)
        self.decoder = GNNDecoder(nodes=nodes, nfeat=nfeat, nhead=nhead)

    def forward(self, x,adj):
        encoder,_,_ = self.encoder(x,adj)
        decoder = self.decoder(encoder)
        return decoder


class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
