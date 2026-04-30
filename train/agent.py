import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch
from collections import deque
from network import DQN

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity: self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx
        return leaf_idx, self.tree[leaf_idx], self.data[leaf_idx - self.capacity + 1]

    @property
    def total_priority(self):
        return self.tree[0]
    
class Agent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.episodes = 100000
        self.gamma = 0.99
        self.lr = 5e-5
        self.lr_min = 1e-5
        self.batch_size = 256
        self.eps = 1.0
        self.eps_min = 0.01
        self.subtrahend = (self.eps - self.eps_min) / (self.episodes * 0.8)
        self.tau = 0.005
        self.max_norm = 1.0

        self.memory = SumTree(100000)
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = (1.0 - self.beta) / (self.episodes * 0.9)
        self.abs_err_upper = 1.0

        self.q_network = DQN().to(self.device)
        self.target_network = DQN().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.episodes, eta_min=self.lr_min)

        self.q_history = deque(maxlen=1000)

    def update_eps(self):
        if self.eps > self.eps_min:
            self.eps = max(self.eps_min, self.eps - self.subtrahend)

    def act(self, state, mask):
        if np.random.random() <= self.eps:
            actions = np.where(mask)[0]
            return np.random.choice(actions)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_t)
            mask_t = torch.BoolTensor(mask).to(self.device).unsqueeze(0)
            q_values[~mask_t] = float('-inf')
            action = q_values.argmax(dim=1).item()  # choose the optimal action's index

            self.q_history.append(q_values[0, action].item())

            return action

    def replay(self):
        if self.memory.n_entries < self.batch_size: return 0.0

        indices = []
        batch = []
        priorities = []
        segment = self.memory.total_priority / self.batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.memory.get_leaf(v)
            priorities.append(p)
            indices.append(idx)
            batch.append(data)

        sampling_probabilities = np.array(priorities) / self.memory.total_priority
        is_weights = np.power(self.memory.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()
        is_weights_t = torch.FloatTensor(is_weights).to(self.device)

        states = torch.tensor(np.array([e[0] for e in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([e[1] for e in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor([e[2] for e in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([e[3] for e in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([e[4] for e in batch], dtype=torch.bool, device=self.device)
        next_masks = torch.tensor(np.array([e[5] for e in batch]), dtype=torch.bool, device=self.device)

        q_values = self.q_network(states)
        q_current = q_values.gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q_online = self.q_network(next_states)
            next_q_online[~next_masks] = float('-inf')
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target_network(next_states)
            q_next = next_q_target.gather(1, next_actions).squeeze(1)
            q_next[~next_masks.any(dim=1)] = 0.0
            q_target = rewards + (self.gamma * q_next * (~dones).float())

        td_errors = torch.abs(q_current - q_target).detach().cpu().numpy()
        for i in range(self.batch_size):
            new_p = np.power(td_errors[i] + 1e-5, self.alpha)
            self.memory.update(indices[i], new_p)

        loss = (is_weights_t * F.smooth_l1_loss(q_current, q_target, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.max_norm)
        self.optimizer.step()

        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return loss.item()