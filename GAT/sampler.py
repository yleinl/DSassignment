import torch


def sample_data(dataset, sample_fraction):
    data = dataset[0]
    num_nodes = data.num_nodes
    sample_size = int(num_nodes * sample_fraction)
    sampled_nodes = torch.randperm(num_nodes)[:sample_size]
    sampled_mask = torch.zeros(num_nodes, dtype=torch.bool)
    sampled_mask[sampled_nodes] = True
    edge_index = data.edge_index
    mask = sampled_mask[edge_index[0]] & sampled_mask[edge_index[1]]
    sampled_edge_index = edge_index[:, mask]
    node_mapping = {node.item(): idx for idx, node in enumerate(sampled_nodes)}
    remapped_edge_index = torch.stack([
        torch.tensor([node_mapping[node.item()] for node in sampled_edge_index[0]]),
        torch.tensor([node_mapping[node.item()] for node in sampled_edge_index[1]])
    ], dim=0)
    sampled_data = data.clone()
    sampled_data.x = data.x[sampled_nodes]
    sampled_data.edge_index = remapped_edge_index
    sampled_data.train_mask = data.train_mask[sampled_nodes]
    sampled_data.val_mask = data.val_mask[sampled_nodes]
    sampled_data.test_mask = data.test_mask[sampled_nodes]
    sampled_data.y = data.y[sampled_nodes]
    sampled_data.num_classes = dataset.num_classes

    return sampled_data