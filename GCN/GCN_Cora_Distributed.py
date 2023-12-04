import pickle
import sys
import time
import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import community as community_louvain

from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import utils
from utils import recv_object, send_object, partition_data, partition_data_louvain


from GCNNet import GCNNet
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
        print(dataset[0].num_nodes)
        partitions = partition_data_louvain(dataset, world_size-1)
        # for i, partition in enumerate(partitions):
        #     owned_nodes_count = torch.sum(partition.owned_nodes_mask).item()
        #     redundant_nodes_count = torch.sum(partition.redundant_nodes_mask).item()
        #     print(f"Partition {i} has {[partition.num_nodes]} total nodes.")
        #     print(f"Partition {i} has {owned_nodes_count} owned nodes.")
        #     print(f"Partition {i} has {redundant_nodes_count} redundant nodes.")
        #     edge_index = partition.edge_index
        #     num_edges = edge_index.size(1) // 2
        #     print(f"Partition {i} has {num_edges} edges.")
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
        print(dataset.edge_index)
        # print(rank, dataset.x.shape)
        # nfeat = dataset.num_node_features
        # nhid = 16
        # nclass = dataset.num_classes
        # dropout = 0.5
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model = GCNNet(nfeat, nhid, nclass, dropout).to(device)
        # # failure_simulation(rank, 0.2)
        # data = dataset.to(device)
        # model.load_state_dict(torch.load('model_epoch_1000_Cora.pth'))
        # model.eval()
        #
        # _, pred = model(data).max(dim=1)
        # correct = float (pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        # acc = correct / data.test_mask.sum().item()
        # print('Accuracy: {:.4f}'.format(acc))



if __name__ == "__main__":
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['LOCAL_RANK'])
    main(rank, world_size)
