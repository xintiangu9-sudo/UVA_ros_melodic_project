import numpy as np

class FloodRescueEnvironment:
    def __init__(self, map_size=(100, 100)):
        self.map_size = map_size
        self.water_levels = np.zeros(map_size) # 动态水位分布
        self.water_flows = np.zeros((*map_size, 2)) # 水流矢量场 (u, v)
        self.victim_locations = [] # 被困人员位置
        self.danger_zones = [] # 危险区域（如旋涡、极高水位）
        self.covered_area = np.zeros(map_size) # 搜索覆盖图
        
    def update_flood_dynamics(self, time_step):
        """模拟动态水位和水流变化 (环境约束)"""
        # 简化模拟：随着时间推移，某些区域水位上升
        self.water_levels += np.random.normal(0, 0.01, self.map_size)
        self.water_levels = np.clip(self.water_levels, 0, 10) # 水位限制在0-10米
        
    def get_environmental_constraints(self, uav_pos):
        """获取无人机当前位置的洪涝环境阻力/约束"""
        x, y = int(uav_pos[0]), int(uav_pos[1])
        if 0 <= x < self.map_size[0] and 0 <= y < self.map_size[1]:
            water_level = self.water_levels[x, y]
            flow_vector = self.water_flows[x, y]
            return water_level, flow_vector
        return 0, np.array([0, 0])

    def calculate_reward(self, uav1_pos, uav2_pos, uav1_action, uav2_action, found_victim):
        """改进的奖励函数：融入搜索覆盖率与协同增益"""
        reward1, reward2 = 0.0, 0.0
        
        # 1. 基础避障惩罚 (基于文献4)
        # 此处省略具体碰撞检测代码，假设通过环境传感器获取
        collision_penalty = -10.0
        
        # 2. 搜索覆盖率奖励 (改进项)
        # 将无人机当前位置标记为已搜索
        def update_coverage(pos):
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < self.map_size[0] and 0 <= y < self.map_size[1]:
                if self.covered_area[x, y] == 0:
                    self.covered_area[x, y] = 1
                    return 1.0 # 新覆盖区域给予奖励
            return 0.0
        
        reward1 += update_coverage(uav1_pos) * 5.0
        reward2 += update_coverage(uav2_pos) * 5.0

        # 3. 协同增益 (改进项)
        # 约束双机距离：不能太近（碰撞/冗余），不能太远（失去通信或协同配合）
        dist = np.linalg.norm(np.array(uav1_pos) - np.array(uav2_pos))
        optimal_dist = 20.0 
        coop_gain = -0.1 * abs(dist - optimal_dist) # 距离最优距离越近，惩罚越小/奖励越高
        reward1 += coop_gain
        reward2 += coop_gain
        
        # 4. 洪涝环境惩罚 (危险水域/逆流)
        level1, _ = self.get_environmental_constraints(uav1_pos)
        if level1 > 8.0: reward1 -= 2.0 # 危险区域惩罚

        # 发现目标重奖
        if found_victim:
            reward1 += 50.0
            reward2 += 50.0

        return reward1, reward2