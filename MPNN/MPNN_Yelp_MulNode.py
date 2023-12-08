import torch
import torch.distributed as dist
import os
import numpy as np
from torch_geometric.datasets import Planetoid, Yelp
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_self_loops
import torch.nn.functional as F
from utils import recv_object, send_object, partition_data_louvain as partition_data, partition_data_louvain_sampled
from sampler import sample_data

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
    def __init__(self, nfeat, nhid, nclass, dropout, rank, world_size):
        super(MPNNNet, self).__init__()

        self.rank = rank

        self.world_size = world_size

        self.nfeat = nfeat

        self.nhid = nhid

        self.mpnn1 = MPNNLayer(nfeat, nhid)

        self.mpnn2 = MPNNLayer(nhid, nclass)

        self.dropout = dropout

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

        print('Range')
        print(Range)
        for node in communication_nodes:
            for i in range(num_partitions):
                if node in Range[i]:
                    requested_nodes_list[i + 1].append(node)

        print('requested_nodes_list')
        print(requested_nodes_list)

        return requested_nodes_list

    def remap_index(self, requested_nodes_list, owned_nodes):
        return requested_nodes_list % owned_nodes.shape[0]

    def forward(self, data):
        num_nodes, x, prev_edge_index, owned_nodes = data.num_nodes, data.x, data.prev_edge_index, data.owned_nodes
        sent_nodes = data.sent_nodes
        partition_size = data.partition_size
        node_partition_id = data.node_partition_id
        edge_index = data.edge_index

        for target_partition in range(1, world_size+1):
            sent_partition_nodes = []
            for edge in prev_edge_index.t():
                for node_idx in edge:
                    node = node_idx.item()
                    if node in owned_nodes and [node % partition_size, target_partition] not in sent_partition_nodes:
                        other_node = edge[1] if node_idx == edge[0] else edge[0]
                        if other_node not in owned_nodes and node_partition_id[other_node] == target_partition:
                            sent_partition_nodes.append([node % partition_size, target_partition])
            sent_nodes[self.rank].append(sent_partition_nodes)

        x = self.mpnn1(x, edge_index)

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
        # x = torch.cat((x[:num_nodes], requested_nodes_feature.reshape(-1, self.nhid)), dim=0)

        replacement = requested_nodes_feature.reshape(-1, self.nhid)
        replacement_length = min(len(replacement), len(x) - len(owned_nodes))
        x[len(owned_nodes):len(owned_nodes) + replacement_length] = replacement[:replacement_length]
        x = self.mpnn2(x, edge_index)
        return F.log_softmax(x, dim=1)[:num_nodes]


def get_master_addr(node_list):
    return '10.141.0.{}'.format(int(node_list[5:8]))


def main(rank, world_size, host_addr_full):
    torch.distributed.init_process_group(backend="gloo", init_method=host_addr_full, rank=rank, world_size=world_size)
    print("Hello, I am ", rank)
    if rank == 0:
        name_data = 'Yelp'
        dataset = Yelp(root='/tmp/' + name_data)
        dataset = sample_data(dataset, sample_fraction=0.01)
        new_data, partitions = partition_data_louvain_sampled(dataset, world_size)
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
        model = MPNNNet(nfeat, nhid, nclass, dropout, rank, world_size).to(device)
        data = dataset.to(device)
        # model.load_state_dict(torch.load('model_epoch_3000_Yelp.pth'))
        model.load_state_dict(torch.load('model_epoch_3000_Yelp.pth', map_location=torch.device('cpu')))

        model.eval()
        _, pred = model(data).max(dim=1)
        pred = torch.sigmoid(model(data)) > 0.5
        pred = pred[:num_nodes]
        correct = pred[data.test_mask].eq(data.y[data.test_mask].to(torch.bool)).sum().item()
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
        correct = final_pred[new_data.test_mask].eq(new_data.y[new_data.test_mask].to(torch.bool)).sum().item()
        acc = correct / (new_data.test_mask.sum().item() * new_data.y.size(1))
        print('Overall new_data: {:.4f}'.format(acc))
    else:
        dataset = recv_object(src=0)
        print("data received on node {} from node 0".format(rank))

        num_nodes = dataset.owned_nodes.shape[0]
        nfeat = dataset.num_node_features
        nhid = 16
        nclass = dataset.num_classes
        dropout = 0.5
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MPNNNet(nfeat, nhid, nclass, dropout, rank, world_size).to(device)
        data = dataset.to(device)
        # model.load_state_dict(torch.load('model_epoch_3000_Yelp.pth'))
        model.load_state_dict(torch.load('model_epoch_3000_Yelp.pth', map_location=torch.device('cpu')))

        model.eval()
        _, pred = model(data).max(dim=1)

        pred = torch.sigmoid(model(data)) > 0.5
        pred = pred[:num_nodes]
        send_object(pred, 0)
        correct = pred[data.test_mask].eq(data.y[data.test_mask].to(torch.bool)).sum().item()
        if data.test_mask.sum().item() != 0:
            acc = correct / (data.test_mask.sum().item() * data.y.size(1))
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