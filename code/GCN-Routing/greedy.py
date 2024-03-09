# 贪心算法实现，计时
from data import *
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import time
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
print(len(edge_features_flat))
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
print("输入的数组:", route)
# 给route前和后都添加一位len(locations)
# 在列表前面插入值
route.insert(0, len(locations)+1)
# 在列表后面插入值
route.append(len(locations)+1)
# 构建边和边的距离字典
edges_dict = {(source, target): feature for source, target, feature in zip(edge[0], edge[1], edge_features_flat)}
# 定义贪心算法
def greedy_algorithm(route, edges_dict):
    total_distance = 0
    path = []
    path.append(route[-1])
    current_point = route[0]  # 从起点开始

    for category in route[1:]:
        possible_edges = [(source, target) for source, target in zip(edge[0], edge[1]) if source == current_point and target in locations_with_category[category]]

        if not possible_edges:
            print(f"No available edges for category {category}")
            continue

        min_edge = min(possible_edges, key=lambda edge_1: edges_dict[edge_1])
        total_distance += edges_dict[min_edge]
        current_point = min_edge[1]
        path.append(current_point)

    # 返回到起点
    total_distance += edges_dict.get((current_point, route[-1]), 0)
    path.append(route[-1])

    return path, total_distance

# 种类所对应的模型ID
locations_with_category = defaultdict(list)

for i in range(len(Y)):
    locations_with_category[Y[i]].append(i + 1)
locations_with_category[len(locations)+1].append(len(locations)+1)
print("locations_with_category =", locations_with_category)

# 调用贪心算法
# start_time = time.time()
result_path, total_distance = greedy_algorithm(route, edges_dict)
# end_time = time.time()
# # 计算执行时间
# execution_time = end_time - start_time
# print(f"Execution time: {execution_time} seconds")
print("result_path = ", result_path)
print("total_distance =",total_distance)
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
for i in range(len(result_path)-1):
    source, target = result_path[i], result_path[i + 1]
    plt.plot([longitude_list[source-1], longitude_list[target-1]], [latitude_list[source-1], latitude_list[target-1]], color='black', linestyle='-')

# 显示图例，只显示一次
plt.legend(loc='upper right')

# 显示图形
plt.show()



