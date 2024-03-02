import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# 读取数据
data = pd.read_csv('GPU_Performance.csv')

features_to_normalize = ['GPU_RunningScore']

# 初始化归一化器
MinMaxScaler = MinMaxScaler()
StandardScaler = StandardScaler()

# 对指定列进行归一化处理
data[features_to_normalize] = MinMaxScaler.fit_transform(data[features_to_normalize])
data[features_to_normalize] = data[features_to_normalize] + 1

# 保存归一化后的数据到新的CSV文件
data.to_csv('normalized_GPU_Performance.csv', index=False)
# data.to_csv('standardized_GPU_Performance.csv', index=False)

print("Data normalization complete. The normalized data is saved to 'normalized_data.csv'.")
