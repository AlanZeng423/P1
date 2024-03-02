# 动作被设计为一个整数，其中 server_id = action % num_servers 选择服务器，model_action = action // num_servers 决定是部署还是移除模型。
# 状态表示为一个包含所有服务器当前存储使用量（相对于其容量的比例）的列表，这提供了关于环境当前状态的基本信息，可进行扩展。

import random

# 定义 EdgeServer 类，代表边缘计算环境中的单个服务器
class EdgeServer:
    def __init__(self, server_id, storage_capacity):
        # 初始化服务器的基本属性
        self.server_id = server_id  # 服务器的唯一标识符
        self.storage_capacity = storage_capacity  # 服务器的存储容量
        self.current_storage_usage = 0  # 服务器当前的存储使用量
        self.models_deployed = []  # 当前部署在服务器上的模型列表

    def deploy_model(self, model_size):
        # 尝试在服务器上部署一个模型
        if self.current_storage_usage + model_size <= self.storage_capacity:
            # 如果有足够的空间部署模型，则更新服务器状态并返回 True
            self.models_deployed.append(model_size)
            self.current_storage_usage += model_size
            return True
        # 如果没有足够空间，则返回 False
        return False

    def remove_model(self):
        # 尝试从服务器上移除一个模型
        if self.models_deployed:
            # 如果服务器上有模型，则移除最后一个部署的模型并更新服务器状态
            model_size = self.models_deployed.pop()
            self.current_storage_usage -= model_size
            return True
        # 如果服务器上没有模型，返回 False
        return False

# 定义 EdgeComputingEnv 类，模拟边缘计算环境
class EdgeComputingEnv:
    def __init__(self):
        # 初始化环境的基本属性
        self.num_servers = 5  # 环境中的服务器数量
        # 创建指定数量的服务器实例，每个服务器具有相同的存储容量
        self.servers = [EdgeServer(i, 1000) for i in range(self.num_servers)]
        # 定义可用模型的大小列表，用于随机选择模型大小
        self.model_sizes = [10, 20, 30, 40, 50]
        self.current_time = 0  # 当前时间步
        self.max_time_steps = 100  # 环境的最大时间步

    def reset(self):
        # 重置环境到初始状态
        self.current_time = 0
        for server in self.servers:
            server.current_storage_usage = 0
            server.models_deployed = []
        # 返回环境的初始状态
        return self.get_state()

    def step(self, action):
        # 执行一个动作并返回新的状态、奖励和是否结束
        self.current_time += 1
        done = self.current_time >= self.max_time_steps  # 检查是否达到最大时间步

        # 解析动作
        server_id = action % self.num_servers  # 选择服务器
        model_action = action // self.num_servers  # 选择动作类型（部署或移除）

        # 根据动作类型执行相应操作
        if model_action == 0:  # 部署模型
            model_size = random.choice(self.model_sizes)  # 随机选择一个模型大小
            # 尝试部署模型并根据结果给出奖励
            if self.servers[server_id].deploy_model(model_size):
                reward = 10  # 部署成功获得正奖励
            else:
                reward = -10  # 部署失败获得负奖励
        else:  # 移除模型
            # 尝试移除模型并根据结果给出奖励
            if self.servers[server_id].remove_model():
                reward = 5  # 移除成功获得正奖励
            else:
                reward = -5  # 移除失败获得负奖励

        # 获取新的环境状态
        next_state = self.get_state()
        return next_state, reward, done, {}

    def get_state(self):
        # 获取环境的当前状态
        # 这里我们简化地使用每个服务器的当前存储使用率作为状态
        state = [server.current_storage_usage / server.storage_capacity for server in self.servers]
        return state

