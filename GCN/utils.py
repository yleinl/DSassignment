import pickle
import torch
import torch.distributed as dist
from community import community_louvain
from torch_geometric.utils import to_networkx

def partition_data_prev(dataset, num_partitions):
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

        # redundant nodes
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
        partition_data.owned_nodes = owned_nodes
        partition_data.num_nodes = num_nodes

        # 标记拥有的节点和冗余节点
        partition_data.owned_nodes_mask = owned_mask[connected_nodes]
        partition_data.redundant_nodes_mask = ~owned_mask[connected_nodes]

        partitions.append(partition_data)

    return partitions

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

        edge_index = data.edge_index
        connected_edges = owned_mask[edge_index[0]] | owned_mask[edge_index[1]]
        new_index = len(owned_nodes)
        external_node_mapping = {}
        remapped_edge_index = edge_index[:, connected_edges]
        remapped_edge_index[0, :] = torch.tensor([owned_nodes.tolist().index(node.item()) if node.item() in owned_nodes else external_node_mapping.setdefault(node.item(), new_index + len(external_node_mapping)) for node in remapped_edge_index[0]])
        remapped_edge_index[1, :] = torch.tensor([owned_nodes.tolist().index(node.item()) if node.item() in owned_nodes else external_node_mapping.setdefault(node.item(), new_index + len(external_node_mapping)) for node in remapped_edge_index[1]])

        # 拷贝原来的数据特征
        partition_data = data.clone()
        partition_data.x = data.x[owned_nodes]
        partition_data.edge_index = remapped_edge_index
        partition_data.train_mask = data.train_mask[owned_nodes]
        partition_data.val_mask = data.val_mask[owned_nodes]
        partition_data.test_mask = data.test_mask[owned_nodes]
        partition_data.y = data.y[owned_nodes]
        partition_data.num_classes = dataset.num_classes
        partition_data.owned_nodes = owned_nodes
        partition_data.num_nodes = len(owned_nodes)

        partitions.append(partition_data)

    return partitions

def partition_data_louvain(dataset, num_partitions):
    data = dataset[0]
    G = to_networkx(data, to_undirected=True)

    # 使用Louvain算法进行图聚类
    partition = community_louvain.best_partition(G)
    num_nodes = data.num_nodes

    # 平均每个分区节点数
    cluster_nodes = [[] for _ in range(num_partitions)]
    for node, cluster_id in partition.items():
        cluster_nodes[cluster_id % num_partitions].append(node)

    # 对群组进行均衡调整
    avg_size = num_nodes // num_partitions
    for cluster in cluster_nodes:
        while len(cluster) > avg_size:
            # 将多余的节点移到其他群组
            moved_node = cluster.pop()
            target_cluster = min(cluster_nodes, key=len)
            target_cluster.append(moved_node)

    partitions = []
    for i in range(num_partitions):
        owned_nodes = torch.tensor(cluster_nodes[i], dtype=torch.long)

        owned_mask = torch.zeros(num_nodes, dtype=torch.bool)
        owned_mask[owned_nodes] = True

        edge_index = data.edge_index
        connected_edges = owned_mask[edge_index[0]] | owned_mask[edge_index[1]]

        # 分区内拥有的节点
        new_index = len(owned_nodes)
        external_node_mapping = {}
        remapped_edge_index = edge_index[:, connected_edges]
        remapped_edge_index[0, :] = torch.tensor([owned_nodes.tolist().index(node.item()) if node.item() in owned_nodes else external_node_mapping.setdefault(node.item(), new_index + len(external_node_mapping)) for node in remapped_edge_index[0]])
        remapped_edge_index[1, :] = torch.tensor([owned_nodes.tolist().index(node.item()) if node.item() in owned_nodes else external_node_mapping.setdefault(node.item(), new_index + len(external_node_mapping)) for node in remapped_edge_index[1]])
        # 拷贝原来的数据特征
        partition_data = data.clone()
        partition_data.x = data.x[owned_nodes]
        partition_data.edge_index = remapped_edge_index
        partition_data.train_mask = data.train_mask[owned_nodes]
        partition_data.val_mask = data.val_mask[owned_nodes]
        partition_data.test_mask = data.test_mask[owned_nodes]
        partition_data.y = data.y[owned_nodes]
        partition_data.num_classes = dataset.num_classes
        partition_data.owned_nodes = owned_nodes
        partition_data.num_nodes = len(owned_nodes)

        partitions.append(partition_data)

    return partitions

def partition_data_louvain_prev(dataset, num_partitions):
    data = dataset[0]
    G = to_networkx(data, to_undirected=True)

    # 使用Louvain算法进行图聚类
    partition = community_louvain.best_partition(G)
    num_nodes = data.num_nodes

    # 平均每个分区节点数
    cluster_nodes = [[] for _ in range(num_partitions)]
    for node, cluster_id in partition.items():
        cluster_nodes[cluster_id % num_partitions].append(node)

    # 对群组进行均衡调整
    avg_size = num_nodes // num_partitions
    for cluster in cluster_nodes:
        while len(cluster) > avg_size:
            # 将多余的节点移到其他群组
            moved_node = cluster.pop()
            target_cluster = min(cluster_nodes, key=len)
            target_cluster.append(moved_node)

    partitions = []
    for i in range(num_partitions):
        owned_nodes = torch.tensor(cluster_nodes[i], dtype=torch.long)
        owned_mask = torch.zeros(num_nodes, dtype=torch.bool)
        owned_mask[owned_nodes] = True

        edge_index = data.edge_index
        connected_nodes = torch.unique(edge_index[:, owned_mask[edge_index[0]] | owned_mask[edge_index[1]]])
        node_mapping = {node.item(): idx for idx, node in enumerate(connected_nodes)}

        # 标记一下所属分区
        node_partition_ids = torch.full((data.num_nodes,), -1, dtype=torch.long)  # 初始化为 -1
        node_partition_ids[connected_nodes] = i

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
        partition_data.node_partition_ids = node_partition_ids[connected_nodes]

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
