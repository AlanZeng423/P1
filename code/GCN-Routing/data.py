import numpy as np
import math
from torch_geometric.data import Data
import torch
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
# 读取CSV文件
df_server = pd.read_csv('server_data.csv')
df_1 = pd.read_csv('model_1.csv')
df_2 = pd.read_csv('model_2.csv')
df_3 = pd.read_csv('model_3.csv')
df_4 = pd.read_csv('model_4.csv')
df_5 = pd.read_csv('model_5.csv')
df_6 = pd.read_csv('model_6.csv')
df_edge = pd.read_csv('edge.csv')
# 假设CSV文件中的列名分别是'Longitude'和'Latitude'
locations = df_server[['Longitude', 'Latitude']].values.tolist()
# print(locations)
# 显卡型号跑分
server_performence = df_server['ServerPerformance'].tolist()
# print(server_performence)

# 文生文模型评分
model_1 = df_1[['ServerID', 'ModelPerformance']].values.tolist()
model_1_efficiency = df_1[['ModelEfficiency']].values.tolist()
# 文生图模型评分
model_2 = df_2[['ServerID', 'ModelPerformance']].values.tolist()
model_2_efficiency = df_2[['ModelEfficiency']].values.tolist()
# 文生图模型综合评分在model_2矩阵中
# model_2 = []
# for row in model_2_data:
#     avg = (row[1] + row[2]) / 2
#     model_2.append([row[0], avg])
# print(model_2)

# 图生视频，评分随便写的，需要改
model_3 = df_3[['ServerID', 'ModelPerformance']].values.tolist()
model_3_efficiency = df_3[['ModelEfficiency']].values.tolist()
# 文生视频，评分随便写的，需要改
model_4 = df_4[['ServerID', 'ModelPerformance']].values.tolist()
model_4_efficiency = df_4[['ModelEfficiency']].values.tolist()
# 音生文，评分随便写的，需要改
model_5 = df_5[['ServerID', 'ModelPerformance']].values.tolist()
model_5_efficiency = df_5[['ModelEfficiency']].values.tolist()
# 文生音，评分随便写的，需要改
model_6 = df_6[['ServerID', 'ModelPerformance']].values.tolist()
model_6_efficiency = df_6[['ModelEfficiency']].values.tolist()
# 模型效率未知，全部为1
model_efficiency = np.vstack((model_1_efficiency, model_2_efficiency, model_3_efficiency, model_4_efficiency, model_5_efficiency, model_6_efficiency))
# print("model_efficiency=", model_efficiency)
# 最大最小归一化函数
def minmax_normalize(data):
    normalized = [[0, 0] for _ in range(len(data))]

    # 如果只有一行,不做归一化
    if len(data) == 1:
        normalized[0][0] = data[0][0]
        normalized[0][1] = 1
        return normalized

    col = [row[1] for row in data]
    min_val = min(col)
    max_val = max(col)

    for i, row in enumerate(data):
        normalized[i][0] = row[0]
        normalized[i][1] = (row[1] - min_val) / (max_val - min_val) + 1

    return normalized

# 6个评分归一化的n * 2的矩阵
M_1 = minmax_normalize(model_1)
M_2 = minmax_normalize(model_2)
M_3 = minmax_normalize(model_3)
M_4 = minmax_normalize(model_4)
M_5 = minmax_normalize(model_5)
M_6 = minmax_normalize(model_6)
# 垂直拼起来，就是所有模型的序号和评分，该评分就是节点的第四个维度
model_performence = np.vstack((M_1, M_2, M_3, M_4, M_5, M_6))
# print("model_performence=", model_performence)

# 拼成一个16 * 6的矩阵，作为节点矩阵
node_matrix = np.hstack((np.array(locations),
                         np.array(server_performence).reshape(-1,1),
                         model_performence,
                         np.array(model_efficiency).reshape(-1,1) ))
# print("node_matrix=", node_matrix)
# 新矩阵,交换列顺序,这个Node_matrix的后五列就是要的节点矩阵，第一列是序号，0-16
Node_matrix = np.zeros_like(node_matrix)
Node_matrix[:,0] = node_matrix[:,3]
Node_matrix[:,1] = node_matrix[:,0]
Node_matrix[:,2] = node_matrix[:,1]
Node_matrix[:,3] = node_matrix[:,2]
Node_matrix[:,4] = node_matrix[:,4]
Node_matrix[:,5] = node_matrix[:,5]
# print("Node_matrix=", Node_matrix)
# print(type(Node_matrix[0][2]))
# 节点特征向量建模完成

# 接下来建模邻接矩阵
# 根据经纬度计算地球上两点距离的半正矢公式（haversine 公式）
def haversine(lon1, lat1, lon2, lat2):

    # 将经度变成弧度
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    # 地球平均半径,单位为公里
    r = 6371

    return c * r
# 邻接矩阵，GCN表示形式
edge_row_1 = df_edge.iloc[0].values.tolist()
edge_row_2 = df_edge.iloc[1].values.tolist()
# 将 edge_row_1 的所有元素转换为整数
edge_row_1 = [int(x) if isinstance(x, (int, float, str)) else x for x in edge_row_1]

# 将 edge_row_2 的所有元素转换为整数
edge_row_2 = [int(x) if isinstance(x, (int, float, str)) else x for x in edge_row_2]
# 边特征向量（暂定1维）
edge = np.vstack((edge_row_1, edge_row_2))
# print("edge=", edge)
edge_features = []
for i in range(len(edge[0])):
    n1 = edge[0][i] - 1
    n2 = edge[1][i] - 1
    lon1, lat1 = locations[n1][0], locations[n1][1]
    lon2, lat2 = locations[n2][0], locations[n2][1]

    distance = haversine(lon1, lat1, lon2, lat2)
    edge_features.append([distance])

edge_features = np.array(edge_features)

Y = [1,1,1,1,2,2,2,3,3,3,3,4,4,4,5,6]
train_mask = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

G = nx.Graph()

# 将节点和边添加到图中
for i in range(len(edge[0])):
    G.add_edge(edge[0][i] - 1, edge[1][i] - 1)  # 减1以调整节点标识符

# 使用 Spring Layout 进行节点布局
pos = nx.spring_layout(G, seed=42)

# 绘制图形
plt.figure(figsize=(10, 8))
nx.draw_networkx(G, pos, with_labels=True, node_size=400, node_color='skyblue', font_size=8, font_color='black', font_weight='bold')
plt.title("Graph Visualization")
plt.show()

Node_matrix = Node_matrix.astype(np.float64)
edge = edge.astype(np.int64)
edge_features = edge_features.astype(np.float64)
# 创建data
data = Data(x=torch.tensor(Node_matrix, dtype=torch.float),
            edge_index=torch.tensor(edge, dtype=torch.float),
            edge_attr=torch.tensor(edge_features, dtype=torch.float),
            y=torch.tensor(Y,dtype=torch.int),
            train_mask=torch.tensor(train_mask,dtype=torch.bool))
print(data)