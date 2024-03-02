# EdgeServer 类代表边缘计算环境中的单个边缘服务器。包含服务器的基本属性，如服务器ID (server_id) 和存储容量 (storage_capacity)，以及当前存储使用量 (current_storage_usage) 和已部署模型的列表 (models_deployed)。
# deploy_model 方法尝试在服务器上部署一个模型。如果服务器有足够的剩余存储空间来容纳新模型，则模型被添加到 models_deployed 列表中，当前存储使用量相应增加，方法返回 True。如果没有足够的空间，方法返回 False。
# remove_model 方法从服务器上移除最近部署的模型（即 models_deployed 列表中的最后一个元素）。如果服务器上有模型被移除，则更新当前存储使用量并返回 True。如果服务器上没有模型可移除，则返回 False

# 定义 EdgeServer 类，代表边缘计算环境中的一台边缘服务器
class EdgeServer:
    def __init__(self, server_id, storage_capacity):
        # 初始化服务器实例的基本属性
        self.server_id = server_id  # 服务器的唯一标识符
        self.storage_capacity = storage_capacity  # 服务器的总存储容量
        self.current_storage_usage = 0  # 服务器当前的存储使用量，初始为0
        self.models_deployed = []  # 当前部署在服务器上的模型列表，初始为空

    def deploy_model(self, model_size):
        """
        尝试在服务器上部署一个给定大小的模型。
        如果服务器有足够的空间来部署这个模型，则模型被部署成功，存储使用量相应增加。
        
        参数:
        - model_size: 将要部署的模型的大小
        
        返回值:
        - True: 如果模型成功部署
        - False: 如果没有足够的空间来部署模型
        """
        # 检查服务器是否有足够的空间来部署新模型
        if self.current_storage_usage + model_size <= self.storage_capacity:
            # 如果有足够的空间，则部署模型
            self.models_deployed.append(model_size)  # 将模型大小添加到已部署模型列表
            self.current_storage_usage += model_size  # 更新当前存储使用量
            return True
        else:
            # 如果没有足够的空间，则部署失败
            return False

    def remove_model(self):
        """
        从服务器上移除一个模型。本示例中简化处理，移除已部署模型列表中的最后一个模型。
        
        返回值:
        - True: 如果有模型被成功移除
        - False: 如果服务器上没有模型可移除
        """
        # 检查是否有模型可以移除
        if self.models_deployed:
            # 如果有，则移除最后一个部署的模型
            model_size = self.models_deployed.pop()  # 移除并获取最后一个模型的大小
            self.current_storage_usage -= model_size  # 更新当前存储使用量
            return True
        else:
            # 如果没有模型可移除，则返回 False
            return False

