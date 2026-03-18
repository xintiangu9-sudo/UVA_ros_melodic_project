import torch
import torch.nn as nn
from brain_navigation import BrainNavigationModule
from snn_controller import SNNController

class CollaborativeAttention(nn.Module):
    """协同注意力模块：交换关键状态信息"""
    def __init__(self, feature_dim=64):
        super(CollaborativeAttention, self).__init__()
        # 使用多头注意力机制的基础思想处理无人机间的特征交互
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, my_feature, partner_feature):
        """
        通过注意力机制融合队友传来的信息（如发现被困人员、危险区域）
        """
        q = self.query(my_feature)
        k = self.key(partner_feature)
        v = self.value(partner_feature)
        
        # 计算注意力权重
        attention_weights = self.softmax(torch.sum(q * k, dim=-1, keepdim=True))
        
        # 提取有价值的协同上下文
        collaborative_context = attention_weights * v
        return collaborative_context

class DualUAVSystem(nn.Module):
    def __init__(self, nav_feature_dim=64, snn_hidden_dim=64):
        super(DualUAVSystem, self).__init__()
        
        # 为两架无人机实例化独立的导航和控制模块
        self.nav_uav1 = BrainNavigationModule(hidden_dim=nav_feature_dim)
        self.nav_uav2 = BrainNavigationModule(hidden_dim=nav_feature_dim)
        
        # SNN控制器的输入维度 = 本机感知障碍物(24) + 本机高层导航特征(64) + 协同上下文(64)
        snn_input_dim = 24 + nav_feature_dim + nav_feature_dim 
        self.snn_uav1 = SNNController(input_dim=snn_input_dim, hidden_dim=snn_hidden_dim)
        self.snn_uav2 = SNNController(input_dim=snn_input_dim, hidden_dim=snn_hidden_dim)
        
        # 共享的协同注意力模块
        self.collab_attention = CollaborativeAttention(feature_dim=nav_feature_dim)

    def forward(self, obs1, obs2):
        """
        obs 字典包含: 'velocity', 'env_features', 'lidar_spikes' (雷达/避障传感器转化为的脉冲)
        """
        # --- 1. 高层认知：位置与网格细胞空间表示 ---
        spatial_rep1 = self.nav_uav1(obs1['velocity'], obs1['env_features'])
        spatial_rep2 = self.nav_uav2(obs2['velocity'], obs2['env_features'])
        
        # --- 2. 任务级协同：交换状态与危险信息 ---
        # UAV1 接收 UAV2 的信息
        context_for_uav1 = self.collab_attention(spatial_rep1, spatial_rep2)
        # UAV2 接收 UAV1 的信息
        context_for_uav2 = self.collab_attention(spatial_rep2, spatial_rep1)
        
        # --- 3. 底层控制：融合感知、导航与协同信息送入 SNN ---
        # 拼接向量：[本机避障感知, 本机导航意图, 队友协同信息]
        snn_input1 = torch.cat([obs1['lidar_spikes'], spatial_rep1, context_for_uav1], dim=-1)
        snn_input2 = torch.cat([obs2['lidar_spikes'], spatial_rep2, context_for_uav2], dim=-1)
        
        # SNN 计算避障与导航动作
        action_uav1 = self.snn_uav1(snn_input1)
        action_uav2 = self.snn_uav2(snn_input2)
        
        return action_uav1, action_uav2

# --- 运行示例 ---
if __name__ == "__main__":
    # 假设 Batch size 为 1
    system = DualUAVSystem()
    
    # 模拟环境输入张量
    mock_obs1 = {
        'velocity': torch.randn(1, 3), 
        'env_features': torch.randn(1, 128),
        'lidar_spikes': torch.rand(1, 24) # 假设24根激光射线
    }
    mock_obs2 = {
        'velocity': torch.randn(1, 3), 
        'env_features': torch.randn(1, 128),
        'lidar_spikes': torch.rand(1, 24)
    }
    
    # 前向推理
    action1, action2 = system(mock_obs1, mock_obs2)
    print("无人机 1 输出动作向量:", action1.detach().numpy())
    print("无人机 2 输出动作向量:", action2.detach().numpy())