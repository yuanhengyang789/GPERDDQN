import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
from collections import deque
import matplotlib.pyplot as plt
import time
from queue import Queue
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import numpy as npt
def generate_map(size=20):
    global start_pos, target_pos
    map_array = npt.zeros((size, size), dtype=npt.float32)  # 初始化为全空地
    start_pos = (size - 1, 0)  # 左下角
    target_pos = (0, size - 1)  # 右上角
    # 示例 1：竖直墙体，开口在第10行
    for i in range(5, 15):
        map_array[i, 10] = 1
    map_array[10, 10] = 0  # 开一个口
    for i in range(1, 10):
        map_array[i, 6] = 1
    map_array[10, 9] = 0  # 开一个口
    # 示例 2：你可以添加其他障碍物
    map_array[8, 5] = 1
    map_array[9, 5] = 1
    map_array[10, 5] = 1
    # 示例 3：你可以添加其他障碍物
    map_array[8, 6] = 1
    map_array[9, 6] = 1
    map_array[10, 6] = 1
    # 示例 4：你可以添加其他障碍物
    map_array[8, 14] = 1
    map_array[9, 14] = 1
    map_array[10, 14] = 1
    map_array[8, 15] = 1
    map_array[9, 15] = 1
    map_array[10, 15] = 1
    # 示例 5：你可以添加其他障碍物
    map_array[16, 3] = 1
    map_array[16, 4] = 1
    map_array[16, 5] = 1
    map_array[15, 3] = 1
    map_array[15, 4] = 1
    map_array[15, 5] = 1
    # 示例 6：你可以添加其他障碍物
    map_array[1, 13] = 1
    map_array[1, 14] = 1
    map_array[1, 15] = 1
    map_array[0, 13] = 1
    map_array[0, 14] = 1
    map_array[0, 15] = 1
    # 示例 7：你可以添加其他障碍物
    map_array[18, 13] = 1
    map_array[18, 14] = 1
    map_array[18, 15] = 1
    map_array[18, 16] = 1
    map_array[19, 13] = 1

    # 示例 8：你可以添加其他障碍物
    map_array[15, 16] = 1
    map_array[15, 17] = 1
    map_array[15, 18] = 1
    map_array[15, 19] = 1
    map_array[14, 16] = 1
    map_array[14, 17] = 1
    map_array[14, 18] = 1
    map_array[14, 19] = 1
    # 示例 9：你可以添加其他障碍物
    map_array[5, 17] = 1
    map_array[5, 18] = 1
    map_array[5, 19] = 1

    # 示例 10：你可以添加其他障碍物
    map_array[5, 0] = 1
    map_array[5, 1] = 1
    # 示例 11：你可以添加其他障碍物
    map_array[12, 0] = 1
    map_array[12, 1] = 1
    map_array[12, 2] = 1
    # 示例 12：你可以添加其他障碍物
    map_array[19, 6] = 1
    map_array[19, 7] = 1
    map_array[19, 8] = 1
    map_array[18, 4] = 1
    map_array[18, 5] = 1
    map_array[18, 6] = 1
    # 示例 13：你可以添加其他障碍物
    map_array[3, 14] = 1
    map_array[4, 14] = 1
    map_array[5, 14] = 1
    # 示例 14：你可以添加其他障碍物
    map_array[17, 10] = 1
    map_array[17, 9] = 1
    map_array[16, 10] = 1
    map_array[16, 9] = 1
    map_array[17, 11] = 1
    map_array[16, 11] = 1
   # 示例 15：你可以添加其他障碍物
    map_array[2, 9] = 1
    map_array[2, 10] = 1
    map_array[2, 11] = 1
    # 示例 16：你可以添加其他障碍物
    map_array[12, 13] = 1
    map_array[12, 12] = 1
    map_array[12, 11] = 1
    # 示例 17：你可以添加其他障碍物
    map_array[9, 18] = 1
    map_array[9, 19] = 1
    # 示例 18：你可以添加其他障碍物
    map_array[2, 2] = 1
    map_array[1, 2] = 1
    map_array[0, 2] = 1
    # 示例 19：你可以添加其他障碍物
    map_array[17, 0] = 1
    map_array[16, 0] = 1
    map_array[10, 0] = 1
    map_array[11, 0] = 1
    map_array[9, 0] = 1
    # 示例 19：你可以添加其他障碍物
    map_array[12, 8] = 1
    map_array[12, 9] = 1
    map_array[12, 5] = 1
    map_array[13, 5] = 1
    map_array[12, 6] = 1
    map_array[13, 6] = 1

    map_array[6, 11] = 1
    map_array[7, 11] = 1
    map_array[8, 11] = 1
    map_array[6, 9] = 1
    map_array[7, 9] = 1
    map_array[8, 9] = 1

    map_array[4, 4] = 1
    map_array[4, 3] = 1
    map_array[5, 4] = 1
    map_array[5, 3] = 1
    return map_array
# 超参数配置
BATCH_SIZE = 64
GAMMA = 0.9
EPS_DECAY = 0.99
MEMORY_SIZE = 10000
LEARNING_RATE = 0.005
NUM_EPISODES = 500
REPLAY_INTERVAL = 20
# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        # 输入20x20，经过两次3x3卷积（无padding，stride=1），输出为32x16x16
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
# 首先添加一个普通的经验回放缓冲区类
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return batch, None, np.ones(batch_size)  # 返回权重全为1的数组
        
    def __len__(self):
        return len(self.memory)

# 定义SumTree
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.size = 0
        self.max_priority = 1.0

    def propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self.propagate(parent, change)

    def retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self.retrieve(left, s)
        else:
            return self.retrieve(right, s - self.tree[left])

    def add(self, priority, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.max_priority = max(self.max_priority, priority)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self.propagate(idx, change)

    def get_leaf(self, value):
        idx = self.retrieve(0, value)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def total_priority(self):
        return max(self.tree[0], 1e-8)  # 确保总优先级不为0

    def get_min_priority(self):
        if self.size == 0:
            return 1.0
        non_zero_priorities = self.tree[self.capacity-1:self.capacity-1+self.size]
        non_zero_priorities = non_zero_priorities[non_zero_priorities > 0]
        if len(non_zero_priorities) == 0:
            return 1.0
        return np.min(non_zero_priorities)

# 定义PrioritizedReplayMemory
class PrioritizedReplayMemoryV1:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_frames = beta_frames
        self.frame_idx = 1
        self.epsilon = 1e-6

    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.tree.max_priority, 1.0)
        
        experience = (state, action, reward, next_state, done)
        self.tree.add(max_priority, experience)

    def sample(self, batch_size, beta=None):
        if beta is None:
            beta = min(1.0, self.beta + self.frame_idx * (1.0 - self.beta) / self.beta_frames)
            self.frame_idx += 1

        if self.tree.size == 0:
            # 如果经验池为空，返回空列表
            return [], [], np.array([])

        batch = []
        indices = []
        priorities = []
        
        segment = self.tree.total_priority() / batch_size
        
        # 确保min_prob不会导致除零
        total_priority = self.tree.total_priority()
        min_priority = self.tree.get_min_priority()
        min_prob = min_priority / total_priority if total_priority > 0 else 1.0

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            
            if a == b:  # 处理边界情况
                b = a + 1e-8
                
            value = np.random.uniform(a, b)
            
            index, priority, data = self.tree.get_leaf(value)
            
            # 确保prob不会导致除零或无效值
            prob = priority / total_priority if total_priority > 0 else 1.0
            
            # 计算权重前确保除数不为零
            if min_prob > 0:
                weight = (prob / min_prob) ** (-beta)
            else:
                weight = 1.0
            
            batch.append(data)
            indices.append(index)
            priorities.append(weight)  # 直接存储计算好的权重

        # 不再需要额外的权重计算
        weights = np.array(priorities, dtype=np.float32)
        weights = weights / weights.max() if weights.max() > 0 else weights

        return batch, indices, weights

    def update_priorities(self, indices, priorities):
        priorities = np.power(priorities + self.epsilon, self.alpha)
        for idx, priority in zip(indices, priorities):
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.size
# 定义双经验池类
class DualPrioritizedReplayMemory:
    def __init__(self, normal_capacity, elite_capacity, alpha=0.7, elite_threshold=0, p0=0.4, p1=0.5, beta_t=0.4):
        self.normal_memory = PrioritizedReplayMemoryV1(normal_capacity, alpha)
        self.elite_memory = PrioritizedReplayMemoryV1(elite_capacity, alpha)
        self.elite_threshold = elite_threshold
        self.normal_ratio = 0.5  # 初始采样比例
        self.alpha = alpha
        self.last_indices_type = None

        # 动态调整相关属性（保留必要参数）
        self.min_ratio = 0.3  # 最小采样比例
        self.max_ratio = 0.8  # 最大采样比例

        # 动态采样参数
        self.p0 = p0  # 初始采样概率
        self.p1 = p1  # 动态采样概率
        self.beta_t = beta_t  # 辅助训练阶段阈值
        self.total_episodes = NUM_EPISODES  # 总训练轮数
        self.current_episode = 0
        self.epsilon_t = 1.0  # 初始化衰减因子

        # 用于计算平均损失
        self.normal_losses = []
        self.elite_losses = []

    def push(self, state, action, reward, next_state, done):
        # 根据奖励决定存入哪个池
        if reward >= self.elite_threshold:
            self.elite_memory.push(state, action, reward, next_state, done)
        else:
            self.normal_memory.push(state, action, reward, next_state, done)

        # 每隔固定episodes调整采样比例
        if done:
            self.current_episode += 1
            if self.current_episode % 10 == 0:
                self.adjust_sampling_ratio()

    def adjust_sampling_ratio(self):
        # 计算当前阶段
        t = self.current_episode / self.total_episodes

        # 更新衰减因子
        self.epsilon_t = max(0.01, self.epsilon_t * 0.995)  # 缓慢衰减

        # 计算平均损失
        l0 = np.mean(self.normal_losses) if self.normal_losses else 0
        l1 = np.mean(self.elite_losses) if self.elite_losses else 0
        total_loss = l0 + l1 if (l0 + l1) > 0 else 1  # 避免除零

        if t < self.beta_t:  # 在辅助训练阶段
            self.normal_ratio = (self.p0 * self.epsilon_t +
                                 self.p1 * (l0 / total_loss))
        else:  # 在主训练阶段
            self.normal_ratio = 0.4  # 或其他固定值

        # 确保采样比例在合理范围内
        self.normal_ratio = max(self.min_ratio, min(self.max_ratio, self.normal_ratio))

        # 清空损失记录
        self.normal_losses = []
        self.elite_losses = []

    def get_memory_stats(self):
        return {
            'normal_size': len(self.normal_memory),
            'elite_size': len(self.elite_memory),
            'normal_ratio': self.normal_ratio
        }

    def sample(self, batch_size, beta=0.4):
        # 如果两个池都是空的，返回空列表
        if self.normal_memory.tree.size == 0 and self.elite_memory.tree.size == 0:
            self.last_indices_type = "empty"
            return [], [], np.array([])
        
        # 根据比例从两个经验池中抽样
        normal_size = int(batch_size * self.normal_ratio)
        elite_size = batch_size - normal_size
        
        # 确保两个池子都有足够的样本
        if (self.normal_memory.tree.size < normal_size) and (self.elite_memory.tree.size < elite_size):
            normal_size = self.normal_memory.tree.size
            elite_size = self.elite_memory.tree.size
        
        if self.normal_memory.tree.size < normal_size:
            normal_size = self.normal_memory.tree.size
            elite_size = min(batch_size - normal_size, self.elite_memory.tree.size)
        
        if self.elite_memory.tree.size < elite_size:
            elite_size = self.elite_memory.tree.size
            normal_size = min(batch_size - elite_size, self.normal_memory.tree.size)
        
        # 如果一个池为空，则从另一个池中抽取全部样本
        if normal_size == 0:
            self.last_indices_type = "elite_only"
            return self.elite_memory.sample(elite_size, beta)
        
        if elite_size == 0:
            self.last_indices_type = "normal_only"
            return self.normal_memory.sample(normal_size, beta)
        
        # 从两个池中抽样
        normal_batch, normal_indices, normal_weights = self.normal_memory.sample(normal_size, beta)
        elite_batch, elite_indices, elite_weights = self.elite_memory.sample(elite_size, beta)
        
        # 合并样本
        batch = normal_batch + elite_batch
        
        # 使用两个独立的索引列表
        self.last_normal_indices = normal_indices
        self.last_elite_indices = elite_indices
        self.last_indices_type = "mixed"
        
        # 合并权重
        weights = np.concatenate((normal_weights, elite_weights))
        
        return batch, (normal_indices, elite_indices), weights

    def update_priorities(self, indices, priorities):
        if self.last_indices_type == "empty":
            return
        
        if self.last_indices_type == "normal_only":
            self.normal_memory.update_priorities(indices, priorities)
            self.normal_losses.extend(priorities)  # 记录损失
            return
        
        if self.last_indices_type == "elite_only":
            self.elite_memory.update_priorities(indices, priorities)
            self.elite_losses.extend(priorities)  # 记录损失
            return
        
        if self.last_indices_type == "mixed":
            normal_indices, elite_indices = indices
            normal_size = len(normal_indices)
            
            # 分别更新两个池的优先级
            if len(normal_indices) > 0:
                normal_priorities = priorities[:normal_size]
                self.normal_memory.update_priorities(normal_indices, normal_priorities)
                self.normal_losses.extend(normal_priorities)  # 记录损失
                
            if len(elite_indices) > 0:
                elite_priorities = priorities[normal_size:]
                self.elite_memory.update_priorities(elite_indices, elite_priorities)
                self.elite_losses.extend(elite_priorities)  # 记录损失
    
    def __len__(self):
        return self.normal_memory.tree.size + self.elite_memory.tree.size

def initialize_q_values(map, target_pos):
    rows, cols = map.shape
    q_values = np.zeros((rows, cols, 8))  # 8个动作
    dr_dc = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 8个动作
    for r in range(rows):
        for c in range(cols):
            if map[r, c] == 0:  # 仅对可通行的格子初始化 Q 值
                for action in range(8):
                    nr, nc = r + dr_dc[action][0], c + dr_dc[action][1]
                    if 0 <= nr < rows and 0 <= nc < cols and map[nr, nc] == 0:
                        dist = np.sqrt((nr - target_pos[0]) ** 2 + (nc - target_pos[1]) ** 2)
                        q_values[r, c, action] = np.exp(-dist)
                    else:
                        q_values[r, c, action] = 0
    return q_values

def initialize_network_weights(net, map_array, target_pos):
    """改进：根据动作类型初始化Q值"""
    q_values = initialize_q_values(map_array, target_pos)
    rows, cols = map_array.shape
    for r in range(rows):
        for c in range(cols):
            state = matrix_to_img((r, c), map_array)
            if map_array[r, c] == 0:
                with torch.no_grad():
                    q_outputs = net(state)
                    for action in range(8):
                        q_outputs[0, action] = q_values[r, c, action]
                    net.fc2.bias.data = q_outputs[0]
# 共用函数
def matrix_to_img(pos, map_array):
    row, col = pos
    state = map_array.copy()
    state[row, col] = 2
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

#贪婪策略选择动作函数
def choose_action(state, policy_net, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 7)
    else:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1).item()
  
#软更细机制
def soft_update(target_net, policy_net, tau):
    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)
#测试函数
def test_net(policy_net, current_pos, target_pos, step_func):
    current_pos = start_pos
    path = [current_pos]  # 记录路径
    prev_action = None
    prev_actions = []
    visited_positions = {}
    for _ in range(100):  # 最多允许100步
        state = matrix_to_img(current_pos, map).to(device)
        with torch.no_grad():
            action = policy_net(state).max(1)[1].item()

        # 根据step_func类型决定参数
        if step_func.__name__ == "step_v3":
            result = step_func(current_pos, action, target_pos, visited_positions, prev_action, prev_actions)
            if isinstance(result, tuple) and len(result) == 5:
                next_pos, reward, done, visited_positions, prev_actions = result
            else:
                next_pos, reward, done = result
        else:
            result = step_func(current_pos, action, target_pos, visited_positions, prev_action)
            if isinstance(result, tuple) and len(result) == 4:
                next_pos, reward, done, visited_positions = result
            else:
                next_pos, reward, done = result

        path.append(next_pos)
        if done:
            break
        prev_action = action
        current_pos = next_pos
    return path

def plot_paths_four(path1, path2, path3, title, start_pos, target_pos, map):
    plt.figure(figsize=(10, 10))
    plt.imshow(map, cmap='gray_r', origin='lower')

    # 路径平移，避免重叠
    offset = 0.1
    # 红色路径向左上偏移
    plt.plot([p[1]-offset for p in path1], [p[0]-offset for p in path1], 'r-', 
             label='G-DPER-DDQN', linewidth=1.5)

    # 绘制起点和终点
    plt.scatter(start_pos[1], start_pos[0], c='blue', s=200, label='Start')
    plt.scatter(target_pos[1], target_pos[0], c='red', s=200, label='Target')

    # 获取当前坐标轴
    ax = plt.gca()
    
    # 设置网格线（保留）
    ax.set_xticks(np.arange(0, map.shape[1], 1)-0.5)
    ax.set_yticks(np.arange(0, map.shape[0], 1)-0.5)
    ax.grid(color='black', linestyle='-', linewidth=0.5, alpha=0.3)  # 保留网格线
    
    # 隐藏刻度标签（1, 2, 3等数字）
    ax.set_xticklabels([])  # 隐藏x轴数字
    ax.set_yticklabels([])  # 隐藏y轴数字
    
    # 取消刻度线，但保留网格线
    ax.tick_params(axis='both', which='both', length=0)  # 不显示刻度线

    plt.title(title)
    plt.legend()
    plt.show()

def smooth_path(path, k=3, num_points=100):

    if len(path) < 4:
        return path
    
    # 提取路径点的x和y坐标
    x = [p[1] for p in path]
    y = [p[0] for p in path]
    
    # 生成参数化的点
    t = np.linspace(0, 1, len(x))
    
    # 创建B样条对象
    try:
        # x方向的B样条
        tck_x = interpolate.splrep(t, x, k=min(k, len(x)-1))
        # y方向的B样条
        tck_y = interpolate.splrep(t, y, k=min(k, len(y)-1))
        
        # 生成平滑路径点
        t_new = np.linspace(0, 1, num_points)
        x_smooth = interpolate.splev(t_new, tck_x)
        y_smooth = interpolate.splev(t_new, tck_y)
        
        # 将平滑后的坐标点组合成路径
        smooth_path = list(zip(y_smooth, x_smooth))
        return smooth_path
    except:
        return path

def is_valid(pos, size):
    r, c = pos
    return 0 <= r < size and 0 <= c < size

def step_v1(current_pos, action, target_pos, visited_positions=None, prev_action=None):
    if visited_positions is None:
        visited_positions = {}
    row, col = current_pos
    actions = {
        0: (-1, 0),   # 上
        1: (1, 0),    # 下
        2: (0, -1),   # 左
        3: (0, 1),    # 右
        4: (-1, -1),  # 左上
        5: (-1, 1),   # 右上
        6: (1, -1),   # 左下
        7: (1, 1)     # 右下
    }
    dr, dc = actions[action]
    size = map.shape[0]

    # Diagonal move obstacle check
    if action >= 4:
        adj_pos1 = (row + dr, col)
        adj_pos2 = (row, col + dc)
        if (is_valid(adj_pos1, size) and map[adj_pos1] == 1) or \
           (is_valid(adj_pos2, size) and map[adj_pos2] == 1):
            reward = -5
            return (row, col), reward, False, visited_positions

    n_row = max(0, min(size - 1, row + dr))
    n_col = max(0, min(size - 1, col + dc))
    done = False
    base_reward = -1
    new_pos = (n_row, n_col)
    # 计算欧几里得距离
    prev_euclidean_distance = np.sqrt((row - target_pos[0]) ** 2 + (col - target_pos[1]) ** 2)
    current_euclidean_distance = np.sqrt((n_row - target_pos[0]) ** 2 + (n_col - target_pos[1]) ** 2)
    distance_reward = 10 * (prev_euclidean_distance - current_euclidean_distance)

    # 记录访问次数并计算奖励
    if new_pos not in visited_positions:
        visited_positions[new_pos] = 1
        repeat_penalty = 0
    else:
        visited_positions[new_pos] += 1
        repeat_penalty = -3

    # 综合所有奖励
    reward = distance_reward + repeat_penalty + base_reward

    if (n_row, n_col) == (row, col) or map[n_row, n_col] == 1:
        reward = -5
        return (row, col), reward, done, visited_positions
    
    if (n_row, n_col) == target_pos:
        reward = 50
        done = True
        return (n_row, n_col), reward, done, visited_positions
    return (n_row, n_col), reward, done, visited_positions

# 修改优化模型函数，适应算法1双经验池
def optimize_model_dual(policy_net, target_net, optimizer, memory, beta=0.4):
    if len(memory) < BATCH_SIZE:
        return
    batch, indices, weights = memory.sample(BATCH_SIZE, beta)
    
    if not batch:  # 如果抽样为空则返回
        return
    # 添加类型检查
    for item in batch:
        if not isinstance(item, tuple) or len(item) != 5:
            print(f"警告：发现无效的样本格式: {item}")
            return
    try:
        state_batch = torch.cat([item[0].to(device) for item in batch])
        action_batch = torch.tensor([item[1] for item in batch], device=device).unsqueeze(1)
        reward_batch = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=device)

        non_final_mask = torch.tensor([item[3] is not None for item in batch], 
                                    device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([item[3] for item in batch 
                                         if item[3] is not None]).to(device)

        policy_next_q_values = torch.zeros(len(batch), device=device)
        with torch.no_grad():
            if len(non_final_next_states) > 0:
                selected_actions = policy_net(non_final_next_states).max(1)[1]
                policy_next_q_values[non_final_mask] = target_net(non_final_next_states).gather(
                    1, selected_actions.unsqueeze(1)
                ).squeeze()

        target_q_values = reward_batch + (GAMMA * policy_next_q_values)
        current_q_values = policy_net(state_batch).gather(1, action_batch)

        is_weights = torch.tensor(weights, device=device, dtype=torch.float32)
        loss = (is_weights * F.mse_loss(current_q_values.squeeze(), target_q_values, reduction='none')).mean()

        priorities = (torch.abs(current_q_values.squeeze() - target_q_values) + 1e-5).detach().cpu().numpy()
        memory.update_priorities(indices, priorities)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
        optimizer.step()
    except Exception as e:
        print(f"优化过程中出错: {e}")
        # 继续训练而不中断

def run_algorithm_v1():
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    # 使用预训练值初始化网络
    initialize_network_weights(policy_net, map, target_pos)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    epsilon = 0.5
    eps_decay = 0.99
    min_epsilon = 0.05
    normal_capacity = int(MEMORY_SIZE * 0.6)
    elite_capacity = MEMORY_SIZE - normal_capacity
    memory = DualPrioritizedReplayMemory(normal_capacity, elite_capacity)
    steps_done = 0
    episode_steps = []
    total_rewards = []
    cumulative_times = []
    cumulative_time = 0
    learning_rates = []
    epsilons = []
    losses = []
    for episode in range(NUM_EPISODES):
        episode_start_time = time.time()
        current_pos = start_pos
        total_reward = 0
        step_count = 0
        visited_positions = {}
        prev_action = None
        episode_loss = 0
        loss_count = 0
        while True:
            state = matrix_to_img(current_pos, map).to(device)
            action = choose_action(state, policy_net, epsilon) 
            next_pos, reward, done, visited_positions = step_v1(
                current_pos, action, target_pos, visited_positions, prev_action)
            next_state = matrix_to_img(next_pos, map).to(device) if not done else None
            memory.push(state, action, reward, next_state, done)
            prev_action = action

            if steps_done % REPLAY_INTERVAL == 0 and len(memory) >= BATCH_SIZE:
                batch, indices, weights = memory.sample(BATCH_SIZE, beta=0.4)
                if batch:
                    state_batch = torch.cat([item[0].to(device) for item in batch])
                    action_batch = torch.tensor([item[1] for item in batch], device=device).unsqueeze(1)
                    reward_batch = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=device)
                    non_final_mask = torch.tensor([item[3] is not None for item in batch], device=device, dtype=torch.bool)
                    non_final_next_states = torch.cat([item[3] for item in batch if item[3] is not None]).to(device)
                    current_q_values = policy_net(state_batch).gather(1, action_batch)
                    next_q_values = torch.zeros(len(batch), device=device)
                    with torch.no_grad():
                        if len(non_final_next_states) > 0:
                            next_actions = policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                            next_q_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_actions).squeeze()
                    target_q_values = reward_batch + (GAMMA * next_q_values)
                    is_weights = torch.tensor(weights, device=device, dtype=torch.float32)
                    loss = (is_weights * F.smooth_l1_loss(current_q_values.squeeze(), target_q_values, reduction='none')).mean()
                    priorities = (torch.abs(current_q_values.squeeze() - target_q_values) + 1e-5).detach().cpu().numpy()
                    memory.update_priorities(indices, priorities)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
                    optimizer.step()
                    episode_loss += loss.item()
                    loss_count += 1
                    soft_update(target_net, policy_net, tau=0.01)
            current_pos = next_pos
            step_count += 1
            steps_done += 1
            total_reward += reward

            if done or step_count >= 2000:
                episode_steps.append(step_count)
                total_rewards.append(total_reward)
                break
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        if episode < 50:
            epsilon = epsilon
        else:
            epsilon = max(min_epsilon, epsilon * eps_decay)
        epsilons.append(epsilon)
        episode_time = time.time() - episode_start_time
        cumulative_time += episode_time
        cumulative_times.append(cumulative_time)
        # 记录平均损失
        avg_loss = episode_loss / loss_count if loss_count > 0 else 0
        losses.append(avg_loss)
        if episode % 1 == 0:
            stats = memory.get_memory_stats()
            print(f'Algorithm 1 - Episode {episode}, Steps: {step_count}, '
                  f'Reward: {total_reward:.1f}, '
                  f'Elite/Normal: {stats["elite_size"]}/{stats["normal_size"]}, '
                  f'Sampling Ratio: {stats["normal_ratio"]:.2f}/{1-stats["normal_ratio"]:.2f}, '
                  f'Epsilon: {epsilon:.3f}, LR: {current_lr:.6f}, Loss: {avg_loss:.6f}')
    final_path = test_net(policy_net, start_pos, target_pos, step_v1)
    return episode_steps, total_rewards, cumulative_times, final_path, learning_rates, policy_net, epsilons

def main():
    # 生成20×20地图
    global map, start_pos, target_pos
    map = generate_map(size=20)
    start_pos = (19, 0)
    target_pos = (0, 19)
    # 运行三个算法获取训练好的网络
    print("Running Algorithm 1 (G-DPER-DDQN)...")
    start1 = time.time()
    steps1, rewards1, times1, path1, learning_rates1, net1, epsilons1 = run_algorithm_v1()
    end1 = time.time()
    print(f"算法1运行时间: {end1 - start1:.6f} 秒")
    # 处理奖励值
    rewards1 = np.array(rewards1)
    # 将低于-10000的值设为-10000
    rewards1 = np.clip(rewards1, -8000, None)
    # 绘制图表
    plt.figure(figsize=(15, 5))
    # 步数比较
    plt.subplot(1, 3, 1)
    plt.plot(steps1, label='Alg1: G-DPER-DDQN')
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()
    # 奖励比较（使用处理后的数据）
    plt.subplot(1, 3, 2)
    plt.plot(rewards1, label='Alg1: G-DPER-DDQN')
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    # 累计时间比较
    plt.subplot(1, 3, 3)
    plt.plot(times1, label='Alg1: G-DPER-DDQN')
    plt.title('Cumulative Time')
    plt.xlabel('Episode')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 绘制epsilon变化图
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(epsilons1) + 1), epsilons1, 'b-', label='Epsilon')
    plt.title(' Epsilon Changes')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.legend()
    plt.show()

        # 路径平滑处理
    smooth_path1 = smooth_path(path1)
        # 显示路径对比（使用平滑后的路径）
    plot_paths_four(smooth_path1, None, None,  
                       "Paths of G-DPER-DDQN", 
                       start_pos, target_pos, map)
if __name__ == "__main__":
    main()
