import torch
import torch.nn as nn
import numpy as np

class BrainNavigationModule(nn.Module):
    def __init__(self, hidden_dim=64):
        super(BrainNavigationModule, self).__init__()
        # 网格细胞 (Grid Cells): 处理速度和方向的路径积分
        self.grid_cell_encoder = nn.Linear(3, hidden_dim) # 输入: [v_x, v_y, heading]
        
        # 位置细胞 (Place Cells): 融合视觉/激光雷达等环境特征
        self.place_cell_encoder = nn.Linear(128, hidden_dim) # 假设输入环境特征向量长度为128
        
        # 融合层：形成拓扑认知
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 拓扑地图记忆 (简单实现：使用字典存储已访问的拓扑节点)
        self.topological_map = {} 
        self.current_node_id = 0

    def forward(self, velocity_info, env_features):
        """
        velocity_info: [batch, 3] 包含速度向量和偏航角
        env_features: [batch, 128] 环境感知特征
        """
        # 1. 模拟网格细胞的路径积分响应
        grid_out = torch.relu(self.grid_cell_encoder(velocity_info))
        
        # 2. 模拟位置细胞对特定地标的响应
        place_out = torch.relu(self.place_cell_encoder(env_features))
        
        # 3. 信息融合，生成当前状态的拓扑表示
        fusion_input = torch.cat([grid_out, place_out], dim=-1)
        spatial_representation = self.fusion_layer(fusion_input)
        
        return spatial_representation
        
    def update_map(self, spatial_representation, global_pos):
        """更新拓扑地图用于路径记忆"""
        # 将高维特征与物理坐标绑定，存入拓扑记忆中
        self.topological_map[self.current_node_id] = {
            'feature': spatial_representation.detach(),
            'position': global_pos
        }
        self.current_node_id += 1