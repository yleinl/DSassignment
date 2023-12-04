import torch
import torch.distributed as dist
import os
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_self_loops
import torch.nn.functional as F
from utils import recv_object, send_object, partition_data_louvain as partition_data

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
    def __init__(self,nfeat, nhid, nclass, dropout, rank, world_size):
        super(Net, self).__init__()
        
        self.rank = rank
        
        self.world_size = world_size
        
        self.conv1 = GraphConvolution(nfeat, nhid)

        self.conv2 = GraphConvolution(nhid, nclass)

        self.dropout=dropout
    
    def generate_communication_list(self, num_nodes, nodes_from, owned_nodes):
        communication_nodes = np.unique([node for node in nodes_from if node not in owned_nodes])
        
        requested_nodes_list = []
        for i in range(self.world_size):
            nodes = []
            requested_nodes_list.append(nodes)
        
        num_partitions = self.world_size - 1
        partition_size = num_nodes // num_partitions
        start_idx = [i * partition_size for i in range(num_partitions)]
        end_idx = [(i + 1) * partition_size if i != num_partitions - 1 else num_nodes for i in range(num_partitions)]
        Range = [range(start_idx[i], end_idx[i]) for i in range(num_partitions)]        

        for node in communication_nodes:
            for i in range(num_partitions):
                if node in Range[i]:
                    requested_nodes_list[i + 1].append(node)
        
        # print(requested_nodes_lis)

        return requested_nodes_list

    def remap_index(self, requested_nodes_list, owned_nodes):
        return requested_nodes_list % owned_nodes.shape[0] 

    def forward(self, data):
        num_nodes, x, edge_index, owned_nodes = data.num_nodes, data.x, data.edge_index, data.owned_nodes
        
        requested_nodes_list =  self.generate_communication_list(num_nodes, edge_index[0], owned_nodes)

        req = None
        for i in range(1, self.world_size):
            if(self.rank == i or len(requested_nodes_list[i]) == 0):
                continue
            
            if self.rank == 1:
                # Send the tensor to process 1
                send_object(torch.tensor(requested_nodes_list[i]), dst = 2)
                print('Rank 1 has requested data')
                requested_nodes_feature = recv_object(src = 2)
                print('Rank1 has received', requested_nodes_feature)
            else:
                # Receive tensor from process 0
                requested_nodes_list = recv_object(src = 1)
                print('Rank 2 has received', requested_nodes_list)
                remapped_idx = self.remap_index(requested_nodes_list, owned_nodes)
                print('Rank 2 index remapped to ', remapped_idx)
                send_object(x[remapped_idx], dst = 1)
                print('Rank2 has sent requested data')

            print('Rank ', self.rank, ' has data ', owned_nodes) 
            print(owned_nodes.shape, owned_nodes.shape[0])
            # print('Rank ', self.rank, ' has data ', requested_nodes_list)

        x = self.conv1(x, edge_index)

        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

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
            print("data sent to node {}".format(dst_rank))
        dataset = partitions[0]
    else:
        dataset = recv_object(src=0)
        print("data received on node {} from node 0".format(rank))
        
        nfeat = dataset.num_node_features
        nhid = 16
        nclass = dataset.num_classes
        dropout = 0.5
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Net(nfeat, nhid, nclass, dropout, rank, world_size).to(device)
        data = dataset.to(device)
        model.load_state_dict(torch.load('model_epoch_1000_Cora.pth'))
        model.eval()
        _, pred = model(data).max(dim=1)
        correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        acc = correct / data.test_mask.sum().item()
        print('Accuracy: {:.4f}'.format(acc))

if __name__ == "__main__":
    rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    world_size = int(os.environ['SLURM_NTASKS'])

    host_addr = get_master_addr(os.environ['SLURM_STEP_NODELIST'])
    port = 1234
    host_addr_full = 'tcp://' + host_addr + ':' + str(port)

    main(rank, world_size, host_addr_full)
