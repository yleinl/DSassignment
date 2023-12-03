import torch
import torch.nn.functional as F

from torch_geometric.nn import GATConv
from torch_geometric.datasets import Yelp

import warnings

from sampler import sample_data

warnings.filterwarnings("ignore")

# Seed for reproducible numbers
torch.manual_seed(2020)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset used Cora
name_data = 'Yelp'
dataset = Yelp(root='/tmp/' + name_data)
dataset = sample_data(dataset, sample_fraction=0.6)

print(f"Number of Classes in {name_data}:", dataset.num_classes)
print(f"Number of Node Features in {name_data}:", dataset.num_node_features)


# Model Definition
class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.hid = 8
        self.in_head = 8
        self.out_head = 1
        
        self.conv1 = GATConv(dataset.num_features, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hid*self.in_head, dataset.num_classes, concat=False, heads=self.out_head, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)


# Train
model = GAT().to(device)
data = dataset.to(device)

# Adam Optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
# loss_fn = torch.nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss for multi-label classification

# Training Loop
# model.train()
# for epoch in range(3000):
#     optimizer.zero_grad()
#     out = model(data)
#     loss = loss_fn(out[data.train_mask], data.y[data.train_mask].to(torch.float))
#     loss.backward()
#     optimizer.step()
#     if (epoch+1)%1000 == 0:
#         print(loss)
#         torch.save(model.state_dict(), f'model_epoch_{epoch+1}_{name_data}.pth')
model.load_state_dict(torch.load('model_epoch_3000_Yelp.pth'))
# Evaluation
model.eval()
pred = torch.sigmoid(model(data)) > 0.5
correct = pred[data.test_mask].eq(data.y[data.test_mask].to(torch.bool)).sum().item()
accuracy = correct / (data.test_mask.sum().item() * data.y.size(1))
print('Accuracy: {:.4f}'.format(accuracy))