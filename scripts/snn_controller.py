import torch
import torch.nn as nn

class LIFNeuron(nn.Module):
    """简化的 Leaky Integrate-and-Fire (LIF) 神经元模型"""
    def __init__(self, threshold=1.0, decay=0.8):
        super(LIFNeuron, self).__init__()
        self.threshold = threshold
        self.decay = decay
        self.mem = 0.0 # 膜电位

    def forward(self, x):
        self.mem = self.mem * self.decay + x
        # 生成脉冲 (使用 Heaviside 阶跃函数，实际训练中需用 Surrogate Gradient 替代)
        spike = (self.mem >= self.threshold).float() 
        # 发放脉冲后膜电位重置
        self.mem = self.mem * (1 - spike)
        return spike

    def reset(self):
        self.mem = 0.0

class SNNController(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, action_dim=4):
        super(SNNController, self).__init__()
        # SNN 全连接层映射
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = LIFNeuron()
        
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.lif2 = LIFNeuron()

    def forward(self, sensor_spikes, time_steps=10):
        """
        多时间步的 SNN 前向传播
        sensor_spikes: [batch, input_dim] 由传感器读数转换成的泊松脉冲序列或直接编码电流
        """
        batch_size = sensor_spikes.size(0)
        action_spikes_accumulator = torch.zeros(batch_size, self.fc2.out_features)
        
        self.lif1.reset()
        self.lif2.reset()

        # 在时间窗口内模拟脉冲传递
        for t in range(time_steps):
            # 将输入转化为电流
            current1 = self.fc1(sensor_spikes)
            spike1 = self.lif1(current1)
            
            current2 = self.fc2(spike1)
            spike2 = self.lif2(current2)
            
            # 累加输出层脉冲，用于解码动作（如脉冲频率编码）
            action_spikes_accumulator += spike2
            
        # 根据脉冲发放频率决定动作强度 (如：上下左右的加速度)
        action = action_spikes_accumulator / time_steps 
        return action