import torch
import torch.nn as nn

class SurrogateSpike(torch.autograd.Function):
    """代理梯度函数：前向传播输出真实脉冲，反向传播使用平滑函数的梯度"""
    @staticmethod
    def forward(ctx, mem, threshold):
        ctx.save_for_backward(mem, threshold)
        # 前向传播：膜电位超过阈值发放脉冲 1，否则为 0
        return (mem >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        mem, threshold = ctx.saved_tensors
        # 反向传播：使用 Fast Sigmoid 作为代理梯度计算
        alpha = 2.0 # 控制梯度的陡峭程度
        grad = grad_output / (alpha * torch.abs(mem - threshold) + 1.0) ** 2
        return grad, None # threshold 不需要梯度

# 实例化一个代理梯度发射器
spike_fn = SurrogateSpike.apply

class LIFNeuron(nn.Module):
    def __init__(self, threshold=1.0, decay=0.8):
        super(LIFNeuron, self).__init__()
        self.threshold = torch.tensor(threshold)
        self.decay = decay
        self.mem = 0.0 

    def forward(self, x):
        self.mem = self.mem * self.decay + x
        # 使用代理梯度函数生成脉冲
        spike = spike_fn(self.mem, self.threshold)
        # 软重置 (Soft Reset) 膜电位，保留梯度传播路径
        self.mem = self.mem - spike * self.threshold
        return spike

    def reset(self):
        self.mem = 0.0

class DualUAVSystem(nn.Module):
    """双无人机 SNN 控制系统"""
    def __init__(self):
        super(DualUAVSystem, self).__init__()
        # 输入：24维雷达数据 + 3维速度数据 = 27维
        input_dim = 24 + 3
        hidden_dim = 64
        # 输出：4维动作均值 (vx, vy, vz, vyaw)
        output_dim = 4 

        # UAV 1 的网络
        self.fc1_uav1 = nn.Linear(input_dim, hidden_dim)
        self.lif1_uav1 = LIFNeuron()
        self.fc2_uav1 = nn.Linear(hidden_dim, output_dim)

        # UAV 2 的网络
        self.fc1_uav2 = nn.Linear(input_dim, hidden_dim)
        self.lif1_uav2 = LIFNeuron()
        self.fc2_uav2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, obs1, obs2):
        # 提取并拼接 UAV 1 的状态
        x1 = torch.cat([obs1['lidar_spikes'], obs1['velocity']], dim=1)
        x1 = self.fc1_uav1(x1)
        s1 = self.lif1_uav1(x1)
        out1 = self.fc2_uav1(s1)

        # 提取并拼接 UAV 2 的状态
        x2 = torch.cat([obs2['lidar_spikes'], obs2['velocity']], dim=1)
        x2 = self.fc1_uav2(x2)
        s2 = self.lif1_uav2(x2)
        out2 = self.fc2_uav2(s2)

        return out1, out2

    def reset_states(self):
        """每个回合结束时，必须重置神经元膜电位"""
        self.lif1_uav1.reset()
        self.lif1_uav2.reset()