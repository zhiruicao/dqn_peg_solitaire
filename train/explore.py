import numpy as np
from collections import Counter
from environment import PegSolitaire

env = PegSolitaire()

def explore_1():
    results = []

    for _ in range(100000):
        env.reset(0,1)
        done = False
        while not done:
            mask = env.get_mask() 
            possible_actions = np.where(mask)[0] 
            if len(possible_actions) == 0:
                break
            action = np.random.choice(possible_actions) 
            _, _, done, _ = env.step(action) 
        results.append(env.remain)

    distribution = Counter(results)
    for count in sorted(distribution.keys()):
        print(f"Remain {count:<2} | {distribution[count]}")

def explore_2():
    results = {}

    for s in range(1, 32):
        actuals = []

        for _ in range(100000):
            env.reset(0,0)
            actual_steps = 0
            
            for _ in range(s):
                rev_mask = (env.board[env.moves[:, 0]] == 0) & (env.board[env.moves[:, 1]] == 0) & (env.board[env.moves[:, 2]] == 1)
                possible_idx = np.where(rev_mask)[0]
                if len(possible_idx) == 0:
                    break
                move_idx = np.random.choice(possible_idx)
                env.board[env.moves[move_idx, [0, 1]]] = 1
                env.board[env.moves[move_idx, 2]] = 0
                actual_steps += 1

            actuals.append(actual_steps)
        
        avg = np.mean(actuals)
        success_rate = np.mean([1 if a == s else 0 for a in actuals])
        results[s] = (avg, success_rate)
        print(f"Target: {s:2d} | Avg: {avg:5.2f} | Success: {success_rate*100:6.2f}%")