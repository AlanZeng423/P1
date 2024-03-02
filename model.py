import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from torch.nn import Linear

class EdgeComputingGNN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(EdgeComputingGNN, self).__init__(aggr='mean')  # "mean" aggregation.
        # Layer to transform node features
        self.lin = Linear(in_channels, out_channels)
        # Layer to transform updated node features to final task predictions
        self.task_lin = Linear(out_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, batch=batch)

    def message(self, x_j, edge_index, size):
        # x_j has shape [E, out_channels]

        # Step 3: Normalize node features.
        row, col = edge_index
        deg = torch_geometric.utils.degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j

    def update(self, aggr_out, batch):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        aggr_out = F.relu(aggr_out)
        
        # Pooling: Combine node embeddings in each graph to a graph embedding
        aggr_out = torch.cat([gmp(aggr_out, batch), gap(aggr_out, batch)], dim=1)
        
        # Apply a final linear layer for task prediction
        task_out = self.task_lin(aggr_out)
        
        return F.log_softmax(task_out, dim=1)

class EdgeServerNetwork(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(EdgeServerNetwork, self).__init__()
        self.conv1 = EdgeComputingGNN(num_features, 128)
        self.conv2 = EdgeComputingGNN(256, num_classes)  # Assuming 256 due to concat of gap and gmp

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index, batch)
        x = F.relu(x)
        x = self.conv2(x, edge_index, batch)
        return x
