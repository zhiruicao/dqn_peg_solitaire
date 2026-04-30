import numpy as np
import torch
from environment import PegSolitaire
from network import DQN
from train import set_seed

class TestAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.q_network = DQN().to(self.device)
        self.q_network.load_state_dict(torch.load('model.pth', map_location=self.device))
        self.q_network.eval() 
    def act(self, state, mask):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_t).cpu().numpy()[0]
            
            actions = np.where(mask)[0]
            if len(actions) == 0:
                return None, 0

            best_idx = np.argmax(q_values[actions])
            action = actions[best_idx]
            
            sorted_q = np.sort(q_values)[::-1]
            chosen_q = q_values[action]
            rank = np.where(sorted_q == chosen_q)[0][0]
            
            return action, rank
        
def test(step, mode):
    wins = 0
    episodes = 1000
    remains = []
    q_ranks = []
        
    for _ in range(episodes):
        state = env.reset(step, mode)
        mask = env.get_mask()
            
        game_ranks = []
            
        while not env.done:
            action, rank = agent.act(state, mask)
            game_ranks.append(rank)
                
            state, _, _, mask = env.step(action)

        if env.remain == 1 and env.board[24] == 1: 
            wins += 1
        remains.append(env.remain)
        if len(game_ranks) > 0:
            q_ranks.append(np.mean(game_ranks))
            
    if mode == 1:
        step = 31 - step
        
    print(f'Step {step:<3}  |  Win {wins/episodes:<6.1%}  |  Remain {np.mean(remains):<5.2f}  |  Q Rank {np.mean(q_ranks):.1f}')

set_seed(0)

env = PegSolitaire()
agent = TestAgent()

print('Reverse Test')
for step in range(1, 30):
    test(step, 0)

print('Forward Test')
for step in range(1, 20):
    test(step, 1)