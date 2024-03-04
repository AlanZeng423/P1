import numpy as np
import math
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
# 16个服务器的经纬度数据
import pandas as pd

# 读取CSV文件
df_1 = pd.read_csv('server_model_data(1).csv')
df = df_1.iloc[0:16]
# 假设CSV文件中的列名分别是'Longitude'和'Latitude'
# 请根据实际情况调整列名
locations = df[['Longitude', 'Latitude']].values.tolist()
# locations = [
#     [144.97476, -37.81517],
#     [144.95256, -37.81524],
#     [144.9712, -37.81239],
#     [144.96918, -37.81679],
#     [144.95692, -37.81808],
#     [144.957909, -37.811269],
#     [144.96893, -37.809081],
#     [144.962656, -37.815461],
#     [144.955026, -37.81285],
#     [144.962313, -37.816356],
#     [144.957211, -37.818071],
#     [144.954367, -37.818202],
#     [144.971971, -37.814225],
#     [144.955905, -37.815975],
#     [144.96521, -37.81132],
#     [144.95123, -37.81761]
# ]
# print(locations)
# 显卡型号跑分
server_performence = df['ServerPerformance'].tolist()
# server_performence = [
#     2.0,
#     2.0,
#     2.0,
#     1.684116016707655,
#     1.684116016707655,
#     1.684116016707655,
#     1.4737479017839716,
#     1.4737479017839716,
#     1.4737479017839716,
#     1.2827028926103758,
#     1.2827028926103758,
#     1.2827028926103758,
#     1.1115275012686887,
#     1.1115275012686887,
#     1.0,
#     1.0
# ]
# print(server_performence)

# 文生文模型评分
df_top4 = df.head(4)

# 将'ServerID'和'ModelPerformance'两列的前四行数据提取到model_1列表中
# 每个元素是[ServerID, ModelPerformance]的形式
model_1 = df_top4[['ServerID', 'ModelPerformance']].values.tolist()
# model_1 = [
#     [1, 72.2],
#     [2, 79.1],
#     [3, 79.4],
#     [4, 76.6]
# ]
# print(type(model_1[0][1]))
# 文生图模型评分
df_rows5to7 = df.iloc[4:7]
model_2 = df_rows5to7[['ServerID', 'ModelPerformance']].values.tolist()
# model_2_data = [
#     [5, 0.865, 0.871],
#     [6, 0.828, 0.891],
#     [7, 0.835, 0.576]
# ]
# 文生图模型综合评分在model_2矩阵中
# model_2 = []
# for row in model_2_data:
#     avg = (row[1] + row[2]) / 2
#     model_2.append([row[0], avg])
# print(model_2)

# 图生视频，评分随便写的，需要改
df_rows8to11 = df.iloc[7:11]
model_3 = df_rows8to11[['ServerID', 'ModelPerformance']].values.tolist()
# model_3 = [
#     [8, 90],
#     [9, 80],
#     [10, 70],
#     [11, 60]
# ]
# 文生视频，评分随便写的，需要改
df_rows12to14 = df.iloc[11:14]
model_4 = df_rows12to14[['ServerID', 'ModelPerformance']].values.tolist()
# model_4 = [
#     [12, 90],
#     [13, 80],
#     [14, 70]
# ]
# 音生文，评分随便写的，需要改
df_rows15to15 = df.iloc[14:15]
model_5 = df_rows15to15[['ServerID', 'ModelPerformance']].values.tolist()
# model_5 = [
#     [15, 90]
# ]
# 文生音，评分随便写的，需要改
df_rows16to16 = df.iloc[15:16]
model_6 = df_rows16to16[['ServerID', 'ModelPerformance']].values.tolist()
# model_6 = [
#     [16, 90]
# ]

# 模型效率未知，全部为1
model_efficiency = df['ModelEfficiency'].tolist()
# print(type(model_efficiency[0]))
# model_efficiency = [
#     1,
#     1,
#     1,
#     1,
#     1,
#     1,
#     1,
#     1,
#     1,
#     1,
#     1,
#     1,
#     1,
#     1,
#     1,
#     1
# ]
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

# print(model_performence)

# 拼成一个16 * 6的矩阵，作为节点矩阵
node_matrix = np.hstack((np.array(locations),
                         np.array(server_performence).reshape(-1,1),
                         model_performence,
                         np.array(model_efficiency).reshape(-1,1) ))

# 新矩阵,交换列顺序,这个Node_matrix的后五列就是要的节点矩阵，第一列是序号，0-16
Node_matrix = np.zeros_like(node_matrix)
Node_matrix[:,0] = node_matrix[:,3]
Node_matrix[:,1] = node_matrix[:,0]
Node_matrix[:,2] = node_matrix[:,1]
Node_matrix[:,3] = node_matrix[:,2]
Node_matrix[:,4] = node_matrix[:,4]
Node_matrix[:,5] = node_matrix[:,5]
# print(Node_matrix)
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
# edge = [
#     [1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,15,15,15,15,15,15,15,15,15,15,15,16],
#     [5,6,7,12,13,14,16,5,6,7,12,13,14,16,5,6,7,12,13,14,16,5,6,7,12,13,14,16,8,9,10,11,8,9,10,11,8,9,10,11,1,2,3,4,5,6,7,12,13,14,16,15]
# ]
# print(edge)
# print(len(df_1))
edge_row_20 = df_1.iloc[18].values.tolist()
edge_row_21 = df_1.iloc[19].values.tolist()
# 将 edge_row_20 的所有元素转换为整数
edge_row_20 = [int(x) if isinstance(x, (int, float, str)) else x for x in edge_row_20]

# 将 edge_row_21 的所有元素转换为整数
edge_row_21 = [int(x) if isinstance(x, (int, float, str)) else x for x in edge_row_21]
# print(type(edge_row_20[1]))
# 边特征向量（暂定1维）
edge = np.vstack((edge_row_20, edge_row_21))
# print(edge)
edge_features = []
for i in range(len(edge[0])):
    n1 = edge[0][i] - 1
    n2 = edge[1][i] - 1
    lon1, lat1 = locations[n1][0], locations[n1][1]
    lon2, lat2 = locations[n2][0], locations[n2][1]

    distance = haversine(lon1, lat1, lon2, lat2)
    edge_features.append([distance])

def Show_Graph(G,color):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False, node_color=color, cmap="Set2")
    plt.show()

edge_features = np.array(edge_features)
# print(type(model_performence[0][1]))
# print(edge)
# print(edge_features)
