import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.datasets import Planetoid

name_data = 'Cora'
dataset = Planetoid(root= '/tmp/' + name_data, name = name_data)
class MPNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MPNNLayer, self).__init__(aggr='mean')  # 'mean' aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        return x_j
class MPNNNet(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MPNNNet, self).__init__()
        self.mpnn1 = MPNNLayer(nfeat, nhid)
        self.mpnn2 = MPNNLayer(nhid, nclass)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.mpnn1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mpnn2(x, edge_index)

        return F.log_softmax(x, dim=1)
nfeat = dataset.num_node_features
nhid = 16
nclass = dataset.num_classes
dropout = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MPNNNet(nfeat, nhid, nclass, dropout).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
model.train()
for epoch in range(1000):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if (epoch+1)%200 == 0:
        print(loss)
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}_{name_data}.pth')
# model.load_state_dict(torch.load('model_epoch_1000_Cora.pth'))
model.eval()
_, pred = model(data).max(dim=1)
correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))
