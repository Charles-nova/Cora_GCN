import torch
import torch.nn.functional as F
# 导入GCN层、GraphSAGE层和GAT层
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

# 加载数据，出错可自行下载，解决方案见下文
dataset = Planetoid(root='./', name='Cora')

class GCN_NET(torch.nn.Module):

    def __init__(self, features, hidden, classes):
        super(GCN_NET, self).__init__()
        self.conv1 = GCNConv(features, hidden)  # shape（输入的节点特征维度 * 中间隐藏层的维度）
        self.conv2 = GCNConv(hidden, classes)  # shaape（中间隐藏层的维度 * 节点类别）

    def forward(self, data):
        # 加载节点特征和邻接关系
        x, edge_index = data.x, data.edge_index
        # 传入卷积层
        x = self.conv1(x, edge_index)
        x = F.relu(x)  # 激活函数
        x = F.dropout(x, training=self.training)  # dropout层，防止过拟合
        x = self.conv2(x, edge_index)  # 第二层卷积层
        # 将经过两层卷积得到的特征输入log_softmax函数得到概率分布
        return F.log_softmax(x, dim=1)


# 判断是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 构建模型，设置中间隐藏层维度为16
model = GCN_NET(dataset.num_node_features, 16, dataset.num_classes).to(device)
# 加载数据
data = dataset[0].to(device)
# 定义优化函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(20):
    optimizer.zero_grad() # 梯度设为零
    out = model(data)  # 模型输出
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])  # 计算损
    loss.backward()  # 反向传播计算梯度
    optimizer.step()  # 一步优化


model.eval()  # 评估模型
_, pred = model(data).max(dim=1)  # 得到模型输出的类别
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())  # 计算正确的个数
acc = correct / int(data.test_mask.sum())  # 得出准确率
print('GCN Accuracy: {:.4f}'.format(acc))