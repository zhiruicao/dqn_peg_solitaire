import numpy as np

class PegSolitaire:
    def __init__(self):
        self.board_layout = np.array([
            [-1, -1,  1,  1,  1, -1, -1],
            [-1, -1,  1,  1,  1, -1, -1],
            [ 1,  1,  1,  1,  1,  1,  1],
            [ 1,  1,  1,  0,  1,  1,  1],
            [ 1,  1,  1,  1,  1,  1,  1],
            [-1, -1,  1,  1,  1, -1, -1],
            [-1, -1,  1,  1,  1, -1, -1]
        ]).flatten()
        
        self.actions = np.where(self.board_layout != -1)[0]  # index list of legal actions
        self.moves = self.get_moves()  # list of action space
        self.reset(0, 1)

    def get_moves(self):
        moves = []
        directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        for f in range(49):
            if self.board_layout[f] == -1: continue
                
            x, y = divmod(f, 7)
            for dx, dy in directions:
                tx, ty = x + dx, y + dy
                if 0 <= tx < 7 and 0 <= ty < 7:
                    t = tx * 7 + ty
                    if self.board_layout[t] != -1:
                        m = (x + dx//2) * 7 + y + dy//2
                        moves.append([f, m, t])
        return np.array(moves, dtype=np.int8)

    def get_mask(self):
        b = self.board
        return (b[self.moves[:, 0]] == 1) & (b[self.moves[:, 1]] == 1) & (b[self.moves[:, 2]] == 0)

    def step(self, action):
        self.board[self.moves[action]] = [0, 0, 1]
        self.remain -= 1

        mask = self.get_mask()
        self.done = not np.any(mask)

        reward = 1.0
        if self.done:
            if self.remain == 1 and self.board[24] == 1:
                reward = 100.0
            else:
                reward = -5.0 * self.remain

        return self.board.copy(), reward, self.done, mask

    def reset(self, steps, mode):
        if mode == 0:
            while True:
                self.board = np.full(49, -1, dtype=np.int8)
                for idx in self.actions: self.board[idx] = 0
                self.board[24] = 1 
                
                actual_steps = 0
                for _ in range(steps):
                    b = self.board
                    rev_mask = (b[self.moves[:, 0]] == 0) & (b[self.moves[:, 1]] == 0) & (b[self.moves[:, 2]] == 1)
                    possible = np.where(rev_mask)[0]
                    if len(possible) == 0: break 
                    idx = np.random.choice(possible)
                    self.board[self.moves[idx, [0, 1, 2]]] = [1, 1, 0]
                    actual_steps += 1
                    
                if actual_steps == steps: 
                    break
        else:
            while True:
                self.board = self.board_layout.copy()
                
                actual_steps = 0
                for _ in range(steps):
                    mask = self.get_mask()
                    actions = np.where(mask)[0]
                    if len(actions) == 0: 
                        break
                    idx = np.random.choice(actions)
                    self.board[self.moves[idx]] = [0, 0, 1]
                    actual_steps += 1
                
                if actual_steps == steps:
                    break

        self.remain = np.sum(self.board == 1)
        self.done = False
        return self.board.copy()