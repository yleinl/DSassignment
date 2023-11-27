#### Imports ####
from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.datasets import Planetoid
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel

#### Loading the Dataset ####
name_data = 'Cora'
dataset = Planetoid(root= '/tmp/' + name_data, name = name_data)

def partition_data(dataset, num_partitions):
    # 基本的数据划分
    data = dataset[0]
    num_nodes = data.num_nodes
    partition_size = num_nodes // num_partitions

    partitions = []
    for i in range(num_partitions):
        start_idx = i * partition_size
        end_idx = (i + 1) * partition_size if i != num_partitions - 1 else num_nodes

        partition_nodes = torch.arange(start_idx, end_idx, dtype=torch.long)

        _, edge_indices = data.edge_index[:, ((data.edge_index[0] >= start_idx) & (data.edge_index[0] < end_idx))].sort(1)
        partition_edges = data.edge_index[:, edge_indices]

        partition_data = data.clone()
        partition_data.x = data.x[partition_nodes]
        partition_data.edge_index = partition_edges

        partitions.append(partition_data)

    return partitions

dataset_partitioned = partition_data(dataset, 3)[0]
dataset_partitioned.num_classes = dataset.num_classes


#### The Graph Convolution Layer ####
class GraphConvolution(MessagePassing):
    def __init__(self, in_channels, out_channels,bias=True, **kwargs):
        super(GraphConvolution, self).__init__(aggr='add', **kwargs)
        self.lin = torch.nn.Linear(in_channels, out_channels,bias=bias)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


class Net(torch.nn.Module):
    def __init__(self,nfeat, nhid, nclass, dropout):
        super(Net, self).__init__()
        self.conv1 = GraphConvolution(nfeat, nhid)

        self.conv2 = GraphConvolution(nhid, nclass)

        self.dropout=dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)

        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


nfeat=dataset.num_node_features
nhid=16
nclass=dataset.num_classes
dropout=0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(nfeat, nhid, nclass, dropout).to(device)
data = dataset[0].to(device)
model.eval()

_, pred = model(data).max(dim=1)
correct = float (pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))
