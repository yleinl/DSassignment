import math
import pickle
import time
import random

import torch
import torch.distributed as dist
from community import community_louvain
from torch_geometric.utils import to_networkx

def partition_data(dataset, num_partitions):
    # data = dataset[0]
    data = dataset
    num_nodes = data.num_nodes
    partition_size = math.ceil(num_nodes / num_partitions)
    node_partition_id = torch.zeros(num_nodes, dtype=torch.long)
    for i in range(len(node_partition_id)):
        node_partition_id[i] = i / partition_size + 1
    partitions = []
    for i in range(num_partitions):
        start_idx = i * partition_size
        end_idx = (i + 1) * partition_size if i != num_partitions - 1 else num_nodes

        owned_nodes = torch.arange(start_idx, end_idx, dtype=torch.long)
        owned_mask = torch.zeros(num_nodes, dtype=torch.bool)
        owned_mask[owned_nodes] = True

        edge_index = data.edge_index
        connected_edges = owned_mask[edge_index[0]] | owned_mask[edge_index[1]]
        sent_nodes = [[] for _ in range(num_partitions)]

        for target_partition in range(1, num_partitions + 1):
            sent_partition_nodes = []
            for edge in edge_index.t():
                for node_idx in edge:
                    node = node_idx.item()
                    if node in owned_nodes and [node % partition_size, target_partition] not in sent_partition_nodes:
                        other_node = edge[1] if node_idx == edge[0] else edge[0]
                        if other_node not in owned_nodes and node_partition_id[other_node] == target_partition:
                            sent_partition_nodes.append([node % partition_size, target_partition])
            sent_nodes[i].append(sent_partition_nodes)

        external_nodes = set(node.item() for node in edge_index[:, connected_edges].flatten()
                             if node.item() not in owned_nodes)
        external_nodes_sorted = sorted(external_nodes)
        external_node_mapping = {node: i + len(owned_nodes) for i, node in enumerate(external_nodes_sorted)}

        remapped_edge_index = edge_index[:, connected_edges]
        remapped_edge_index[0, :] = torch.tensor([
            owned_nodes.tolist().index(node.item()) if node.item() in owned_nodes
            else external_node_mapping[node.item()]
            for node in remapped_edge_index[0]
        ])
        remapped_edge_index[1, :] = torch.tensor([
            owned_nodes.tolist().index(node.item()) if node.item() in owned_nodes
            else external_node_mapping[node.item()]
            for node in remapped_edge_index[1]
        ])
        external_features = data.x[torch.tensor(external_nodes_sorted, dtype=torch.long)]

        partition_data = data.clone()
        partition_data.x = torch.cat((data.x[owned_nodes], external_features), dim=0)
        partition_data.edge_index = remapped_edge_index
        partition_data.train_mask = data.train_mask[owned_nodes]
        partition_data.node_partition_id = node_partition_id
        partition_data.prev_edge_index = edge_index[:, connected_edges]
        partition_data.val_mask = data.val_mask[owned_nodes]
        partition_data.test_mask = data.test_mask[owned_nodes]
        partition_data.y = data.y[owned_nodes]
        partition_data.num_classes = dataset.num_classes
        partition_data.owned_nodes = owned_nodes
        partition_data.num_nodes = num_nodes
        partition_data.communication_sources = 0
        partition_data.sent_nodes = sent_nodes
        partition_data.partition_size = partition_size

        partitions.append(partition_data)

    return partitions

def remove_edges(G, fraction=0.8):
    edges = list(G.edges())
    num_to_remove = int(len(edges) * fraction)
    edges_to_remove = set(random.sample(edges, num_to_remove))
    G.remove_edges_from(edges_to_remove)
    return G

def remap_data_louvain(data, cluster_nodes):
    new_order = [node for cluster in cluster_nodes for node in cluster]
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(new_order)}

    data.x = data.x[torch.tensor(new_order)]
    data.y = data.y[torch.tensor(new_order)]
    data.train_mask = data.train_mask[torch.tensor(new_order)]
    data.val_mask = data.val_mask[torch.tensor(new_order)]
    data.test_mask = data.test_mask[torch.tensor(new_order)]

    remapped_edge_index = torch.stack([
        torch.tensor([old_to_new[node.item()] for node in edge]) for edge in data.edge_index.t()
    ]).t()
    data.edge_index = remapped_edge_index

    return data

def partition_data_louvain(dataset, num_partitions):
    data = dataset[0]
    G = to_networkx(data, to_undirected=True)
    G = remove_edges(G)

    partition = community_louvain.best_partition(G)
    num_nodes = data.num_nodes

    cluster_nodes = [[] for _ in range(num_partitions)]
    for node, cluster_id in partition.items():
        cluster_nodes[cluster_id % num_partitions].append(node)

    avg_size = num_nodes // num_partitions
    for cluster in cluster_nodes:
        while len(cluster) > avg_size:
            moved_node = cluster.pop()
            target_cluster = min(cluster_nodes, key=len)
            target_cluster.append(moved_node)

    data = remap_data_louvain(data, cluster_nodes)
    data.num_classes = dataset.num_classes

    return data, partition_data(data, num_partitions)

def partition_data_louvain_sampled(dataset, num_partitions):
    # data = dataset[0]
    data = dataset
    G = to_networkx(data, to_undirected=True)
    G = remove_edges(G)

    partition = community_louvain.best_partition(G)
    num_nodes = data.num_nodes

    cluster_nodes = [[] for _ in range(num_partitions)]
    for node, cluster_id in partition.items():
        cluster_nodes[cluster_id % num_partitions].append(node)

    avg_size = num_nodes // num_partitions
    for cluster in cluster_nodes:
        while len(cluster) > avg_size:
            moved_node = cluster.pop()
            target_cluster = min(cluster_nodes, key=len)
            target_cluster.append(moved_node)

    data = remap_data_louvain(data, cluster_nodes)
    data.num_classes = dataset.num_classes

    return data, partition_data(data, num_partitions)

def send_with_timeout(tensor, dst, timeout=3):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            dist.send(tensor, dst=dst)
            return True
        except Exception as e:
            time.sleep(1)  # 等待一段时间后重试
    return False

def recv_with_timeout(tensor, src, timeout=3):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            dist.recv(tensor, src=src)
            return True
        except Exception as e:
            time.sleep(1)
    return False

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


def isend_object(obj, dst):
    buffer = pickle.dumps(obj)
    buffer_tensor = torch.ByteTensor(list(buffer))
    work_data = dist.isend(buffer_tensor, dst=dst)

    return work_data


def send_size_tensor(obj, dst):
    buffer = pickle.dumps(obj)
    buffer_tensor = torch.ByteTensor(list(buffer))
    size_tensor = torch.tensor([buffer_tensor.numel()], dtype=torch.long)
    dist.send(size_tensor, dst=dst)


def irecv_object(src, size_tensor):
    buffer_tensor = torch.empty((size_tensor.item(),), dtype=torch.uint8)
    work_data = dist.irecv(buffer_tensor, src=src)

    return work_data, buffer_tensor


def try_send(tensor, dst, TIMEOUT=1):
    req = dist.isend(tensor=tensor, dst=dst)
    start_time = time.time()
    while time.time() - start_time < TIMEOUT:
        if req.is_completed():
            return True
        time.sleep(0.1)  # 短暂休眠，避免过度占用CPU
    return False


def try_recv(tensor, src, TIMEOUT=1):
    req = dist.irecv(tensor=tensor, src=src)
    start_time = time.time()
    while time.time() - start_time < TIMEOUT:
        if req.is_completed():
            return True
        time.sleep(0.1)
    return False


def wait_with_timeout(request, timeout_seconds=0.5):
    start_time = time.time()
    while not request.is_completed():
        time.sleep(0.1)
        if time.time() - start_time > timeout_seconds:
            print(f"Timeout reached for request. Skipping...")
            return False
    return True

