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
    def __init__(self, nfeat, nhid, nclass, dropout, rank, world_size):
        super(Net, self).__init__()
        
        self.rank = rank
        
        self.world_size = world_size
        
        self.nfeat = nfeat

        self.nhid = nhid

        self.conv1 = GraphConvolution(nfeat, nhid)

        self.conv2 = GraphConvolution(nhid, nclass)

        self.dropout=dropout

    def forward(self, data):
        num_nodes, x, edge_index, owned_nodes = data.num_nodes, data.x, data.prev_edge_index, data.owned_nodes
        communication_sources, sent_nodes = data.communication_sources, data.sent_nodes
        # size_send_requests = []
        # size_recv_buffers = []
        # for target_partition in range(1, world_size):
        #     if target_partition != self.rank:
        #         nodes_to_send = [node_info[0] for node_info in sent_nodes[self.rank - 1][target_partition - 1]]
        #         nodes_to_send = np.unique(nodes_to_send)
        #         size_to_send = torch.tensor(x[nodes_to_send].shape, dtype=torch.int64)
        #         size_send_req = dist.isend(tensor=size_to_send, dst=target_partition)
        #         size_send_requests.append(size_send_req)
        #
        # size_recv_requests = []
        #
        # for source_partition in range(1, world_size):
        #     if source_partition != self.rank:
        #         size_recv_buffer = torch.zeros(2, dtype=torch.int64)
        #         req = dist.irecv(tensor=size_recv_buffer, src=source_partition)
        #         size_recv_requests.append(req)
        #         size_recv_buffers.append((source_partition, size_recv_buffer))
        # for req in size_send_requests:
        #     req.wait()
        # for req in size_recv_requests:
        #     req.wait()
        # recv_sizes = {}
        # for (source_partition, buffer) in size_recv_buffers:
        #     recv_sizes[source_partition] = buffer.tolist()
        # send_requests = []
        # recv_requests = []
        # recv_buffers = []
        #
        # for target_partition in range(1, world_size):
        #     if target_partition != self.rank:
        #         nodes_to_send = [node_info[0] for node_info in sent_nodes[self.rank - 1][target_partition - 1]]
        #         nodes_to_send = np.unique(nodes_to_send)
        #         if len(nodes_to_send) > 0:
        #             tensor_to_send = x[nodes_to_send]
        #             send_req = dist.isend(tensor=tensor_to_send, dst=target_partition)
        #             send_requests.append(send_req)
        #
        # for source_partition in range(1, world_size):
        #     if source_partition != self.rank:
        #         size = recv_sizes[source_partition]
        #         recv_buffer = torch.zeros(size, dtype=x.dtype)
        #         recv_req = dist.irecv(tensor=recv_buffer, src=source_partition)
        #         print(size)
        #         recv_requests.append(recv_req)
        #         recv_buffers.append(recv_buffer)
        #
        # for req in send_requests:
        #     req.wait()
        # for req in recv_requests:
        #     req.wait()
        #
        # requested_nodes_feature = []
        # for buffer in recv_buffers:
        #     requested_nodes_feature.append(buffer)
        # requested_nodes_feature = torch.cat(requested_nodes_feature, dim=0)
        # x = torch.cat((x, requested_nodes_feature.reshape(-1, self.nfeat)), dim=0)
        edge_index = data.edge_index

        x = self.conv1(x, edge_index)

        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = F.relu(x)
        size_send_requests = []
        size_recv_buffers = []
        for target_partition in range(0, world_size):
            if target_partition != self.rank:
                nodes_to_send = [node_info[0] for node_info in sent_nodes[self.rank][target_partition]]
                nodes_to_send = np.unique(nodes_to_send)
                size_to_send = torch.tensor(x[nodes_to_send].shape, dtype=torch.int64)
                size_send_req = dist.isend(tensor=size_to_send, dst=target_partition)
                size_send_requests.append(size_send_req)

        size_recv_requests = []

        for source_partition in range(0, world_size):
            if source_partition != self.rank:
                size_recv_buffer = torch.zeros(2, dtype=torch.int64)
                req = dist.irecv(tensor=size_recv_buffer, src=source_partition)
                size_recv_requests.append(req)
                size_recv_buffers.append((source_partition, size_recv_buffer))
        for req in size_send_requests:
            req.wait()
        for req in size_recv_requests:
            req.wait()
        recv_sizes = {}
        for (source_partition, buffer) in size_recv_buffers:
            recv_sizes[source_partition] = buffer.tolist()
        send_requests = []
        recv_requests = []
        recv_buffers = []

        for target_partition in range(0, world_size):
            if target_partition != self.rank:
                nodes_to_send = [node_info[0] for node_info in sent_nodes[self.rank][target_partition]]
                nodes_to_send = np.unique(nodes_to_send)
                if len(nodes_to_send) > 0:
                    tensor_to_send = x[nodes_to_send]
                    send_req = dist.isend(tensor=tensor_to_send, dst=target_partition)
                    send_requests.append(send_req)

        for source_partition in range(0, world_size):
            if source_partition != self.rank:
                size = recv_sizes[source_partition]
                recv_buffer = torch.zeros(size, dtype=x.dtype)
                recv_req = dist.irecv(tensor=recv_buffer, src=source_partition)
                print(size)
                recv_requests.append(recv_req)
                recv_buffers.append(recv_buffer)

        requested_nodes_feature = []
        for buffer in recv_buffers:
            requested_nodes_feature.append(buffer)
        requested_nodes_feature = torch.cat(requested_nodes_feature, dim=0)
        x = torch.cat((x[:len(owned_nodes)], requested_nodes_feature.reshape(-1, self.nhid)), dim=0)
        
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)[:len(owned_nodes)]

def get_master_addr(node_list):
    return '10.141.0.{}'.format(int(node_list[5:8]))

def main(rank, world_size, host_addr_full):
    torch.distributed.init_process_group(backend="gloo", init_method=host_addr_full, rank=rank, world_size=world_size)
    print("Hello, I am ", rank)
    if rank == 0:
        name_data = 'Cora'
        dataset = Planetoid(root='/tmp/' + name_data, name=name_data)
        new_data, partitions = partition_data(dataset, world_size)
        print(partitions[0].num_nodes)
        for dst_rank in range(1, world_size):
            send_object(partitions[dst_rank], dst=dst_rank)
            print("data sent to node {}".format(dst_rank))

        dataset = partitions[0]
        print("data received on node {} from node 0".format(rank))

        num_nodes = dataset.owned_nodes.shape[0]
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
        pred = pred[:num_nodes]
        correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        if data.test_mask.sum().item() != 0:
            acc = correct / data.test_mask.sum().item()
            print("The accuracy at rank {} is {}".format(rank, acc))
        else:
            print("Rank {} has no test data".format(rank))

        all_pred = []
        all_pred.append(pred)
        for src_rank in range(1, world_size):
            pred = recv_object(src=src_rank)
            all_pred.append(pred)
        final_pred = torch.cat(all_pred, dim=0)
        final_pred = torch.tensor(final_pred)
        correct = float(final_pred[new_data.test_mask].eq(new_data.y[new_data.test_mask]).sum().item())
        acc = correct / new_data.test_mask.sum().item()
        print('Overall Accuracy: {:.4f}'.format(acc))
    else:
        dataset = recv_object(src=0)
        print("data received on node {} from node 0".format(rank))
        
        num_nodes = dataset.owned_nodes.shape[0]
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
        pred = pred[:num_nodes]
        send_object(pred, 0)
        correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        if data.test_mask.sum().item() != 0:
            acc = correct / data.test_mask.sum().item()
            print("The accuracy at rank {} is {}".format(rank, acc))
        else:
            print("Rank {} has no test data".format(rank))


if __name__ == "__main__":
    rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    world_size = int(os.environ['SLURM_NTASKS'])

    host_addr = get_master_addr(os.environ['SLURM_STEP_NODELIST'])
    port = 1234
    host_addr_full = 'tcp://' + host_addr + ':' + str(port)

    main(rank, world_size, host_addr_full)
