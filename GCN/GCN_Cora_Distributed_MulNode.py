import pickle
import torch
import torch.distributed as dist
import os
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_self_loops
import torch.nn.functional as F

def partition_data(dataset, num_partitions):
    data = dataset[0]
    num_nodes = data.num_nodes
    partition_size = num_nodes // num_partitions

    partitions = []
    for i in range(num_partitions):
        # 分区内拥有的节点范围
        start_idx = i * partition_size
        end_idx = (i + 1) * partition_size if i != num_partitions - 1 else num_nodes

        # 分区内拥有的节点
        owned_nodes = torch.arange(start_idx, end_idx, dtype=torch.long)
        owned_mask = torch.zeros(num_nodes, dtype=torch.bool)
        owned_mask[owned_nodes] = True

        # 荣誉节点
        edge_index = data.edge_index
        connected_nodes = torch.unique(edge_index[:, owned_mask[edge_index[0]] | owned_mask[edge_index[1]]])

        node_mapping = {node.item(): idx for idx, node in enumerate(connected_nodes)}

        remapped_edge_index = torch.stack([
            torch.tensor([node_mapping[node.item()] for node in edge_index[0] if node.item() in connected_nodes]),
            torch.tensor([node_mapping[node.item()] for node in edge_index[1] if node.item() in connected_nodes])
        ], dim=0)

        # 拷贝原来的数据特征
        partition_data = data.clone()
        partition_data.x = data.x[connected_nodes]
        partition_data.edge_index = remapped_edge_index
        partition_data.train_mask = data.train_mask[connected_nodes]
        partition_data.val_mask = data.val_mask[connected_nodes]
        partition_data.test_mask = data.test_mask[connected_nodes]
        partition_data.y = data.y[connected_nodes]
        partition_data.num_classes = dataset.num_classes

        # 标记拥有的节点和冗余节点
        partition_data.owned_nodes_mask = owned_mask[connected_nodes]
        partition_data.redundant_nodes_mask = ~owned_mask[connected_nodes]

        partitions.append(partition_data)

    return partitions

def send_object(obj, dst):
    buffer = pickle.dumps(obj)
    buffer_list = list(buffer)
    tensor = torch.ByteTensor(buffer_list)
    size = torch.tensor([tensor.numel()], dtype=torch.long)
    dist.send(size, dst=dst)
    dist.send(tensor, dst=dst)

def recv_object(src):
    size = torch.tensor([0], dtype=torch.long)
    dist.recv(size, src=src)
    buffer = torch.empty((size.item(),), dtype=torch.uint8)
    dist.recv(buffer, src=src)
    obj = pickle.loads(buffer.numpy().tobytes())
    return obj


def get_master_addr(node_list):
    return '10.141.0.{}'.format(int(node_list[5:8]))

def main(rank, world_size, host_addr_full):
    torch.distributed.init_process_group(backend="gloo", init_method=host_addr_full, rank=rank, world_size=world_size)
    print("Hello, I am ", rank)
    if rank == 0:
        name_data = 'Cora'
        dataset = Planetoid(root= '/tmp/' + name_data, name = name_data)
        partitions = partition_data(dataset, world_size-1)
        for dst_rank in range(1, world_size):
            send_object(partitions[dst_rank-1], dst=dst_rank)
            print("send", dst_rank)
    else:
        dataset = recv_object(src=0)
        print("received")
        nfeat = dataset.num_node_features
        nhid = 16
        nclass = dataset.num_classes
        dropout = 0.5
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Net(nfeat, nhid, nclass, dropout).to(device)
        data = dataset.to(device)
        model.load_state_dict(torch.load('model_epoch_1000_Cora.pth'))
        model.eval()

        _, pred = model(data).max(dim=1)
        correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        acc = correct / data.test_mask.sum().item()
        print('Accuracy: {:.4f}'.format(acc))

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

if __name__ == "__main__":
    rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    world_size = int(os.environ['SLURM_NTASKS'])

    host_addr = get_master_addr(os.environ['SLURM_STEP_NODELIST'])
    port = 1234
    host_addr_full = 'tcp://' + host_addr + ':' + str(port)

    main(rank, world_size, host_addr_full)
