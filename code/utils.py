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