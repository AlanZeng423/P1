# 贪心算法实现，计时
from data import *
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import heapq
import time
from collections import defaultdict
# 将列表解析成经度和纬度的两个列表
longitude, latitude = zip(*locations)
# 输出最小的经度和纬度，为假设的用户服务器位置
min_1 = min(longitude)
min_2 = max(latitude)
# 计算[min_1,min_2]到locations中所有坐标的距离
distance = [haversine(min_1, min_2, lon, lat) for lon, lat in zip(longitude, latitude)]
edge = edge.tolist()
edge_features = edge_features.tolist()
# 将[min_1,min_2]到所有其他节点的边添加到 edge 中
for i in range(len(locations)):
    # 可去
    edge[0].append(len(locations)+1)  # 17 号点的索引
    edge[1].append(i + 1)
    # 可回
    edge[0].append(i+1)
    edge[1].append(len(locations)+1)
# 复制每个元素并放在原元素的后面
distance_copy = np.repeat(distance, 2)
# 将相应的距离添加到 edge_features 中
edge_features_flat = [item for sublist in edge_features for item in sublist]
edge_features_flat.extend(distance_copy) # 可以把每条边target点上耽误的时间都算进这条边的距离中，具体计算为这条边对应target点的(c1/server_performence + c2/model_efficiency)加到对应的edge_features_flat中
node_efficiency = []
# 权重，根据测试自己调整
c1 = 1
c2 = 1
for i in range(len(model_efficiency)):
    node_efficiency.append((c1 / server_performence[i]) + (c2 / model_efficiency[i]))
node_efficiency = [float(value) for value in node_efficiency]
for i in range(len(edge[1])):
    if edge[1][i] == 17:
        edge_features_flat[i] += 0
    else:
        edge_features_flat[i] += node_efficiency[edge[1][i]-1]
print("edge =", edge)
print("edge_features_flat =", edge_features_flat)
# 输入需要用到的模型，顺序有效
route = []

print("请输入需要的模型类型，以-1结束:")
while True:
    try:
        num = int(input())
        if num == -1:
            break
        route.append(num)
    except ValueError:
        print("请输入有效的整数.")
route.insert(0, len(locations)+1)
route.append(len(locations)+1)
# 插值构成完整的路径
print("输入的数组:", route)
# route = [17,1,2,3,17]
# locations.append([min_1,min_2])
# node_positions = {i + 1: (lon, lat) for i, (lon, lat) in enumerate(locations)}
# print("node_positions =", node_positions)
Y.append(len(locations)+1)
print(Y)
# Initialize the route_pro array
route_pro = []
# Fill route_pro based on the condition
for r in route:
    indices = [i+1 for i, y in enumerate(Y) if y == r]
    route_pro.append(indices)
print(route_pro)


def create_graph_and_costs(edge, edge_features_flat):
    graph = defaultdict(list)
    costs = {}
    for start, end, cost in zip(edge[0], edge[1], edge_features_flat):
        graph[start].append(end)
        costs[(start, end)] = cost
    return graph, costs

locations.append([min_1,min_2])
def heuristic(node, goal):
    # 简化版的启发式函数，这里假设为0，因为我们的路径受到route_pro的限制
    return haversine(locations[node-1][0],locations[node-1][1],locations[goal-1][0],locations[goal-1][1])


def astar(graph, costs, start, goal):
    open_set = [(0, start)]
    came_from = {}

    g_score = defaultdict(lambda: float('inf'))
    g_score[start] = 0

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(current)
            return path[::-1]

        for neighbor in graph[current]:
            tentative_g_score = g_score[current] + costs.get((current, neighbor), float('inf'))
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))

    return None

# 转换边信息和成本
graph, costs = create_graph_and_costs(edge, edge_features_flat)

# 为了保持路径连续，直接使用route_pro中的点按顺序查找路径
final_path = [route_pro[0][0]]  # 起始节点
for i in range(len(route_pro)-1):
    best_path = []
    best_cost = float('inf')
    for start in route_pro[i]:
        for end in route_pro[i+1]:
            path = astar(graph, costs, start, end)
            if path:
                cost = sum(costs[(path[i], path[i+1])] for i in range(len(path)-1))
                if cost < best_cost:
                    best_path = path
                    best_cost = cost
    if best_path:
        final_path.extend(best_path[1:])  # 避免重复添加节点

print("Final Path:", final_path)

# 使用默认字典收集每个类别的坐标
category_coordinates = defaultdict(list)

for lon, lat, category in zip(longitude, latitude, Y):
    category_coordinates[category].append((lon, lat))

# 绘制散点图
plt.figure(figsize=(10, 8))

# 遍历类别字典，绘制相同类别的坐标使用相同颜色
for category, coordinates in category_coordinates.items():
    color = plt.cm.jet(category / max(Y))  # 使用viridis颜色映射
    plt.scatter(*zip(*coordinates), color=color, s=50, label=f'Category {category}')

# 添加标签和标题
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Scatter Plot of Coordinates')

# 绘制用户点
plt.scatter(min_1, min_2, color='purple', s=100, label='User Coordinate')

# 绘制最短路径
# 将元组转换为列表
longitude_list = list(longitude)
# 将最小经度添加到列表末尾
longitude_list.append(min_1)
# 同理
latitude_list = list(latitude)
latitude_list.append(min_2)
for i in range(len(final_path)-1):
    source, target = final_path[i], final_path[i + 1]
    plt.plot([longitude_list[source-1], longitude_list[target-1]], [latitude_list[source-1], latitude_list[target-1]], color='black', linestyle='-')

# 显示图例，只显示一次
plt.legend(loc='upper right')

# 显示图形
plt.show()