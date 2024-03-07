import torch
from torch_geometric.data import Data
from graphModeling import Node_matrix, edge, edge_features
from torch_geometric.nn import GCNConv
import torch.optim as optim
import torch.nn.functional as F

print(Node_matrix)
print(Node_matrix.dtype)
# node_matrix = node_matrix.astype(np.float64)
# print
# 创建一个Data对象
data = Data(x=torch.tensor(Node_matrix, dtype=torch.float), 
            edge_index=torch.tensor(edge, dtype=torch.float),
            edge_attr=torch.tensor(edge_features, dtype=torch.float))


print(data)



class GCNModel(torch.nn.Module):
    def __init__(self):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(data.num_node_features, 16)
        self.conv2 = GCNConv(16, data.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GCNModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)

