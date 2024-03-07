import torch
import numpy as np
from torch_geometric.data import Data
# from graphModeling import Node_matrix, edge, edge_features
from data import Node_matrix, edge, edge_features
from torch_geometric.nn import GCNConv
import torch.optim as optim
import torch.nn.functional as F
from queue import PriorityQueue

# 数据预处理
Node_matrix = Node_matrix.astype(np.float64)
edge = edge.astype(np.int64)
edge_features = edge_features.astype(np.float64)


# 创建一个Data对象
data = Data(x=torch.tensor(Node_matrix, dtype=torch.float), 
            edge_index=torch.tensor(edge, dtype=torch.long),
            edge_attr=torch.tensor(edge_features, dtype=torch.float))

# 初始data.edge_index是从1开始的，需要减1
data.edge_index = data.edge_index - 1


print(f"!!!{data.edge_index.max()}")
print(f"!!!{data.edge_index.min()}")
print(f"---!!!---{data.edge_index}---!!!---")

print(f"DATA:-----------{data}-------------")

# 定义GNN`GCN`模型

num_node_features = Node_matrix.shape[1]  # 假设Node_matrix的所有列都是节点特征
output_features = 8  # 假定输出特征的维度为8，根据需要调整

class GCNModel(torch.nn.Module):
    def __init__(self, num_node_features, output_features):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)  # 第一个GCN层
        self.conv2 = GCNConv(16, output_features)  # 第二个GCN层，决定输出特征的维度

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))  # 应用ReLU激活函数
        x = F.dropout(x, training=self.training)  # 应用dropout防止过拟合
        x = self.conv2(x, edge_index)  # 第二层GCN
        return x

# 创建模型和优化器
model = GCNModel(num_node_features=num_node_features, output_features=output_features)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

# 训练模型
model.train()

for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    
    # 由于这是一个示例，我们没有真正的目标（labels）来计算损失
    # 在实际应用中，您需要根据任务确定如何计算损失
    # 以下是一个假设的损失计算示例
    loss = loss_fn(out, torch.randn_like(out))  # 假设损失
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss {loss.item()}')


# After training, we can use the model to make predictions
model.eval()  # 将模型设置为评估模式
with torch.no_grad():
    node_embeddings = model(data.x, data.edge_index) #
# node_embeddings是节点的嵌入表示，可以用于后续任务，如节点分类、链接预测等
    


# def heuristic(node, target_node, node_embeddings):
#     # 使用节点表示的欧氏距离作为启发式的估计
#     return torch.norm(node_embeddings[node] - node_embeddings[target_node], p=2).item()

# from queue import PriorityQueue

# def a_star_search(start_node, goal_node, graph, node_embeddings):
#     frontier = PriorityQueue()
#     frontier.put((0, start_node))
#     came_from = {start_node: None}
#     cost_so_far = {start_node: 0}
    
#     while not frontier.empty():
#         current = frontier.get()[1]
        
#         if current == goal_node:
#             break
        
#         for next in graph.neighbors(current):
#             new_cost = cost_so_far[current] + graph[current][next].get('weight', 1)  # Assume default weight=1 if not specified
#             if next not in cost_so_far or new_cost < cost_so_far[next]:
#                 cost_so_far[next] = new_cost
#                 priority = new_cost + heuristic(next, goal_node, node_embeddings)
#                 frontier.put((priority, next))
#                 came_from[next] = current
                
#     # Reconstruct path
#     current = goal_node
#     path = []
#     while current is not None:
#         path.append(current)
#         current = came_from.get(current, None)
#     path.reverse()  # because we followed the path backwards
#     return path
    


def create_adj_list(edge_index, edge_attr):
    adj_list = {}
    for i, (src, dest) in enumerate(edge_index.t()):
        weight = edge_attr[i].item()
        if src.item() not in adj_list:
            adj_list[src.item()] = []
        adj_list[src.item()].append((dest.item(), weight))
    return adj_list


from queue import PriorityQueue

def heuristic(node, goal_node, node_embeddings):
    return torch.norm(node_embeddings[node] - node_embeddings[goal_node], p=2).item()

def a_star_search(start_node, goal_node, adj_list, node_embeddings):
    frontier = PriorityQueue()
    frontier.put((0, start_node))
    came_from = {start_node: None}
    cost_so_far = {start_node: 0}

    while not frontier.empty():
        current = frontier.get()[1]

        if current == goal_node:
            break

        for next_node, weight in adj_list.get(current, []):
            new_cost = cost_so_far[current] + weight
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + heuristic(next_node, goal_node, node_embeddings)
                frontier.put((priority, next_node))
                came_from[next_node] = current

    # Reconstruct path
    current = goal_node
    path = []
    while current is not None:
        path.append(current)
        current = came_from.get(current, None)
    path.reverse()
    return path




adj_list = create_adj_list(data.edge_index, data.edge_attr)
start_node, goal_node = 0, 5  # 示例起点和终点
path = a_star_search(start_node, goal_node, adj_list, node_embeddings)
print(f"Path from {start_node} to {goal_node}: {path}")

