#### Imports ####
from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.datasets import Planetoid
import torch.distributed as dist
import os

#### Loading the Dataset ####
name_data = 'Cora'
dataset = Planetoid(root= '/tmp/' + name_data, name = name_data)


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

        req = None
        if rank == 0:
            # Send the tensor to process 1
            req = dist.isend(tensor = x, dst = 1)
            print('Rank 0 started sending')
        else:
            # Receive tensor from process 0
            req = dist.irecv(tensor = x, src = 0)
            print('Rank 1 started receiving')
        req.wait()
        print('Rank ', rank, ' has data ', x[0])

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

def get_master_addr(node_list):
    return '10.141.0.{}'.format(int(node_list[5:8]))

rank = int(os.environ['SLURM_PROCID'])
local_rank = int(os.environ['SLURM_LOCALID'])
world_size = int(os.environ['SLURM_NTASKS'])

host_addr = get_master_addr(os.environ['SLURM_STEP_NODELIST'])
port = 1234
host_addr_full = 'tcp://' + host_addr + ':' + str(port)
torch.distributed.init_process_group(backend = "gloo", init_method = host_addr_full, rank = rank, world_size = world_size)

nfeat=dataset.num_node_features
nhid=16
nclass=dataset.num_classes
dropout=0.5

#### Training ####
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(nfeat, nhid, nclass, dropout).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

'''
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
'''

model.load_state_dict(torch.load('model_epoch_1000_Cora.pth'))
model.eval()
_, pred = model(data).max(dim=1)
correct = float (pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))
