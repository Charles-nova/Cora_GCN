# import torch
# from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.utils import add_self_loops, degree
#
#
# class GCNConv(MessagePassing):
#     def __init__(self, in_channels, out_channels):
#         super(GCNConv, self).__init__(aggr='add')
#         self.linear = torch.nn.Linear(in_channels, out_channels)
#
#     def forward(self, x, edge_index):
#         # x has shape [N, in_channels]
#         # edge_index has shape [2, E], where E is the number of edges
#         edge_index, _ = add_self_loops(edge_index, num_nodes= x.size(0))  # add self-connections
#         x = self.linear(x)
#         return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
#
#     def message(self, x_j):
#         pass

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt


class GCN_NET3(torch.nn.Module):
    '''
    three-layers GCN
    two-layers GCN has a better performance
    '''
    def __init__(self, num_features, hidden_size1, hidden_size2, classes):
        '''
        :param num_features: each node has a [1,D] feature vector
        :param hidden_size1: the size of the first hidden layer
        :param hidden_size2: the size of the second hidden layer
        :param classes: the number of the classes
        '''
        super(GCN_NET3, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_size1)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)  # use dropout to over ove-fitting
        self.conv2 = GCNConv(hidden_size1, hidden_size2)
        self.conv3 = GCNConv(hidden_size2, classes)
        self.softmax = torch.nn.Softmax(dim=1) # each raw

    def forward(self, Graph):
        x, edge_index = Graph.x, Graph.edge_index
        out = self.conv1(x, edge_index)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out, edge_index)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv3(out, edge_index)
        out = self.softmax(out)
        return out


dataset = Planetoid(root='./', name='Cora')  # if root='./', Planetoid will use local dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # use cpu or gpu
model = GCN_NET3(dataset.num_node_features, 128, 64, dataset.num_classes).to(device)
data = dataset[0].to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)  # define optimizer
num_epoch = 200
L = []  # store loss

model.train()
for epoch in range(num_epoch):
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    _, val_pred = model(data).max(dim=1)
    val_corrent = val_pred[data.val_mask].eq(data.y[data.val_mask]).sum().item()
    val_acc = val_corrent / data.val_mask.sum()
    print('Epoch: {}  loss : {:.4f}  val_acc: {:.4f}'.format(epoch, loss.item(), val_acc.item()))
    L.append(loss.item())
    loss.backward()
    optimizer.step()

model.eval()
_, pred = model(data).max(dim=1)
corrent = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
acc = corrent / data.test_mask.sum()
print("test accuracy is {:.4f}".format(acc.item()))
# plot the curve of loss
n = [i for i in range(num_epoch)]
plt.plot(n, L)
plt.show()

