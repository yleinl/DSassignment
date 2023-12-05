import pickle
import torch
import torch.distributed as dist
from community import community_louvain
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

def partition_data_prev(dataset, num_partitions):
    data = dataset
    num_nodes = data.num_nodes
    partition_size = num_nodes // num_partitions
    node_partition_id = torch.zeros(num_nodes, dtype=torch.long)
    for i in range(len(node_partition_id)):
        node_partition_id[i] = i / partition_size + 1

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
        connected_edges = owned_mask[edge_index[0]] | owned_mask[edge_index[1]]
        connected_nodes = torch.unique(edge_index[:, owned_mask[edge_index[0]] | owned_mask[edge_index[1]]])

        node_mapping = {node.item(): idx for idx, node in enumerate(connected_nodes)}

        communication_sources = []
        sent_nodes = []
        for index in range(1, num_partitions + 1):
            communication_nodes = []
            for node in edge_index[:, connected_edges].view(-1):
                if node not in owned_nodes and node_partition_id[node] == index and node not in communication_nodes:
                    communication_nodes.append(node.item())
            communication_sources.append(communication_nodes)

        for index in range(1, num_partitions + 1):
            sent_partition_nodes = []
            for edge in edge_index.t():
                for node_idx in edge:
                    node = node_idx.item()
                    if node in owned_nodes and node not in sent_partition_nodes:
                        other_node = edge[1] if node_idx == edge[0] else edge[0]
                        if other_node not in owned_nodes and node_partition_id[other_node] == index:
                            sent_partition_nodes.append(node)
                            break
            sent_nodes.append(sent_partition_nodes)

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
        partition_data.node_partition_id = node_partition_id
        partition_data.prev_edge_index = edge_index[:, connected_edges]
        partition_data.test_mask = data.test_mask[connected_nodes]
        partition_data.y = data.y[connected_nodes]
        partition_data.num_classes = dataset.num_classes
        partition_data.owned_nodes = owned_nodes
        partition_data.num_nodes = end_idx - start_idx + 1
        partition_data.communication_sources = communication_sources
        partition_data.sent_nodes = sent_nodes

        # 标记拥有的节点和冗余节点
        partition_data.owned_nodes_mask = owned_mask[connected_nodes]
        partition_data.redundant_nodes_mask = ~owned_mask[connected_nodes]

        partitions.append(partition_data)

    return partitions

def partition_data(dataset, num_partitions):
    # data = dataset[0]
    data = dataset
    num_nodes = data.num_nodes
    partition_size = num_nodes // num_partitions
    node_partition_id = torch.zeros(num_nodes, dtype=torch.long)
    for i in range(len(node_partition_id)):
        node_partition_id[i] = i / partition_size + 1
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
        communication_sources = []
        sent_nodes = []
        for index in range(1, num_partitions + 1):
            communication_nodes = []
            for node in edge_index[:, connected_edges].view(-1):
                if node not in owned_nodes and node_partition_id[node] == index and node not in communication_nodes:
                    communication_nodes.append(node.item())
            communication_sources.append(communication_nodes)


        for index in range(1, num_partitions + 1):
            sent_partition_nodes = []
            for edge in edge_index.t():
                for node_idx in edge:
                    node = node_idx.item()
                    if node in owned_nodes and node not in sent_partition_nodes:
                        other_node = edge[1] if node_idx == edge[0] else edge[0]
                        if other_node not in owned_nodes and node_partition_id[other_node] == index:
                            sent_partition_nodes.append(node)
                            break
            sent_nodes.append(sent_partition_nodes)
        external_node_mapping = {}
        remapped_edge_index = edge_index[:, connected_edges]
        remapped_edge_index[0, :] = torch.tensor([owned_nodes.tolist().index(node.item()) if node.item() in owned_nodes else external_node_mapping.setdefault(node.item(), new_index + len(external_node_mapping)) for node in remapped_edge_index[0]])
        remapped_edge_index[1, :] = torch.tensor([owned_nodes.tolist().index(node.item()) if node.item() in owned_nodes else external_node_mapping.setdefault(node.item(), new_index + len(external_node_mapping)) for node in remapped_edge_index[1]])

        # 拷贝原来的数据特征
        partition_data = data.clone()
        partition_data.x = data.x[owned_nodes]
        partition_data.edge_index = remapped_edge_index
        partition_data.train_mask = data.train_mask[owned_nodes]
        partition_data.node_partition_id = node_partition_id
        partition_data.prev_edge_index = edge_index[:, connected_edges]
        partition_data.val_mask = data.val_mask[owned_nodes]
        partition_data.test_mask = data.test_mask[owned_nodes]
        partition_data.y = data.y[owned_nodes]
        partition_data.num_classes = dataset.num_classes
        partition_data.owned_nodes = owned_nodes
        partition_data.num_nodes = len(owned_nodes)
        partition_data.communication_sources = communication_sources
        partition_data.sent_nodes = sent_nodes

        partitions.append(partition_data)

    return partitions

def remap_data_louvain(data, cluster_nodes):
    # 创建新的节点顺序映射
    new_order = [node for cluster in cluster_nodes for node in cluster]
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(new_order)}

    # 重排节点特征、标签和掩码
    data.x = data.x[torch.tensor(new_order)]
    data.y = data.y[torch.tensor(new_order)]
    data.train_mask = data.train_mask[torch.tensor(new_order)]
    data.val_mask = data.val_mask[torch.tensor(new_order)]
    data.test_mask = data.test_mask[torch.tensor(new_order)]

    # 重排边索引
    remapped_edge_index = torch.stack([
        torch.tensor([old_to_new[node.item()] for node in edge]) for edge in data.edge_index.t()
    ]).t()
    data.edge_index = remapped_edge_index

    return data

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

    data = remap_data_louvain(data, cluster_nodes)
    data.num_classes = dataset.num_classes

    return partition_data(data, num_partitions)

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
    # node_partition_id = torch.zeros(num_nodes, dtype=torch.long)
    # for index, cluster_node in enumerate(cluster_nodes):
    #     for i in cluster_node:
    #         node_partition_id[i] = index + 1

    data = remap_data_louvain(data, cluster_nodes)
    data.num_classes = dataset.num_classes

    return partition_data_prev(data, num_partitions)

    # partitions = []
    # for i in range(num_partitions):
    #     owned_nodes = torch.tensor(cluster_nodes[i], dtype=torch.long)
    #     owned_mask = torch.zeros(num_nodes, dtype=torch.bool)
    #     owned_mask[owned_nodes] = True
    #
    #     edge_index = data.edge_index
    #     connected_nodes = torch.unique(edge_index[:, owned_mask[edge_index[0]] | owned_mask[edge_index[1]]])
    #     node_mapping = {node.item(): idx for idx, node in enumerate(connected_nodes)}
    #
    #     # 标记一下所属分区
    #     node_partition_ids = torch.full((data.num_nodes,), -1, dtype=torch.long)  # 初始化为 -1
    #     node_partition_ids[connected_nodes] = i
    #     connected_edges = owned_mask[edge_index[0]] | owned_mask[edge_index[1]]
    #
    #     remapped_edge_index = torch.stack([
    #         torch.tensor([node_mapping[node.item()] for node in edge_index[0] if node.item() in connected_nodes]),
    #         torch.tensor([node_mapping[node.item()] for node in edge_index[1] if node.item() in connected_nodes])
    #     ], dim=0)
    #
    #     # 拷贝原来的数据特征
    #     partition_data = data.clone()
    #     partition_data.x = data.x[connected_nodes]
    #     partition_data.edge_index = remapped_edge_index
    #     partition_data.train_mask = data.train_mask[connected_nodes]
    #     partition_data.val_mask = data.val_mask[connected_nodes]
    #     partition_data.test_mask = data.test_mask[connected_nodes]
    #     partition_data.y = data.y[connected_nodes]
    #     partition_data.num_classes = dataset.num_classes
    #     partition_data.node_partition_id = node_partition_id
    #     partition_data.prev_edge_index = edge_index[:, connected_edges]
    #
    #     # 标记拥有的节点和冗余节点
    #     partition_data.owned_nodes_mask = owned_mask[connected_nodes]
    #     partition_data.redundant_nodes_mask = ~owned_mask[connected_nodes]
    #     partition_data.node_partition_ids = node_partition_ids[connected_nodes]
    #
    #     partitions.append(partition_data)

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
