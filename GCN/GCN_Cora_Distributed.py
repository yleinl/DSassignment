import pickle
import sys
import time
import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

from torch_geometric.datasets import Planetoid

from GCNNet import GCNNet
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

def send_heartbeat(dst, status_dict, TIMEOUT = 2):
    try:
        req = dist.isend(tensor=torch.zeros(1), dst=dst)
        wait_result = req.wait(timeout=TIMEOUT)
        status_dict[dst] = wait_result
    except:
        status_dict[dst] = False

def check_node_status(world_size):
    node_status = mp.Manager().dict({i: True for i in range(1, world_size)})
    processes = []
    for i in range(1, world_size):
        p = mp.Process(target=send_heartbeat, args=(i, node_status))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    return dict(node_status)

def failure_simulation(rank, rate):
    if random.random() < rate:
        print(f"Node {rank} failed.")
        sys.exit(1)
def main(rank, world_size):
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    if rank == 0:
        HEARTBEAT_INTERVAL = 0.1
        name_data = 'Cora'
        dataset = Planetoid(root= '/tmp/' + name_data, name = name_data)
        partitions = partition_data(dataset, world_size-1)
        for dst_rank in range(1, world_size):
            send_object(partitions[dst_rank-1], dst=dst_rank)
        while True:
            node_status = check_node_status(world_size)

            inactive_nodes = [node for node, status in node_status.items() if not status]
            if len(inactive_nodes) == 0:
                print("All nodes are active.")
            elif len(inactive_nodes) == world_size - 1:
                break
            else:
                print(f"Some nodes failed. Inactive nodes: {inactive_nodes}")
                # 尝试搞得失败处理逻辑，还不合理
                # for node in inactive_nodes:
                #     dataset = partitions[node - 1]
                #     nfeat = dataset.num_node_features
                #     nhid = 16
                #     nclass = dataset.num_classes
                #     dropout = 0.5
                #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                #     model = GCNNet(nfeat, nhid, nclass, dropout).to(device)
                #     data = dataset.to(device)
                #     model.load_state_dict(torch.load('model_epoch_1000_Cora.pth'))
                #     model.eval()
                #     _, pred = model(data).max(dim=1)
                #     correct = float (pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
                #     acc = correct / data.test_mask.sum().item()
                #     print('Accuracy: {:.4f}'.format(acc))


            time.sleep(HEARTBEAT_INTERVAL)

    else :
        dataset = recv_object(src=0)
        nfeat = dataset.num_node_features
        nhid = 16
        nclass = dataset.num_classes
        dropout = 0.5
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GCNNet(nfeat, nhid, nclass, dropout).to(device)
        failure_simulation(rank, 0.2)
        data = dataset.to(device)
        model.load_state_dict(torch.load('model_epoch_1000_Cora.pth'))
        model.eval()

        _, pred = model(data).max(dim=1)
        correct = float (pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        acc = correct / data.test_mask.sum().item()
        print('Accuracy: {:.4f}'.format(acc))



if __name__ == "__main__":
    world_size = 3
    rank = int(os.environ['RANK'])
    main(rank, world_size)
