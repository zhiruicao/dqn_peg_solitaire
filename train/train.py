import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
from environment import PegSolitaire
from agent import Agent

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def draw(eps, rems, sts, lss):
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axs[0].plot(eps, rems, color='tab:red', linewidth=1)
    axs[0].set_ylabel('Remain')
    axs[0].grid(True, linestyle='--', alpha=0.6)

    axs[1].plot(eps, sts, color='tab:blue', linewidth=1)
    axs[1].set_ylabel('Step')
    axs[1].grid(True, linestyle='--', alpha=0.6)

    axs[2].plot(eps, lss, color='tab:green', linewidth=1)
    axs[2].set_ylabel('Loss')
    axs[2].set_xlabel('Episode')
    axs[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('train_curve.png') 
    plt.show()

set_seed(0)

env = PegSolitaire()
agent = Agent()

steps = deque(maxlen=75)
scores = deque(maxlen=75)
losses = deque(maxlen=75)
remains = deque(maxlen=75)
least_rem = 1.5

history_remains = []
history_steps = []
history_losses = []
history_episodes = []

for episode in range(1, agent.episodes + 1):
    progress = episode / agent.episodes
    percent = 0.7
    if progress < percent:
        mode = 0
        max_step = int(3 + 25 * ( (progress / percent) ** 2 ) )
        curr_step = np.random.randint(max(1, max_step - 5), max_step + 1)
        step = curr_step
    else:
        mode = 1
        max_step = int(5 * (1 - (progress - percent) / (1 - percent) ) )
        curr_step = np.random.randint(0, max_step + 1)
        step = 31 - curr_step

    state = env.reset(curr_step, mode)
    mask = env.get_mask()

    total_reward = 0
    done = False
    
    while not done:
        if not np.any(mask): break

        action = agent.act(state, mask)
        next_state, reward, done, next_mask = env.step(action)

        max_p = np.power(agent.abs_err_upper, agent.alpha)
        agent.memory.add(max_p, (state, action, reward, next_state, done, next_mask))

        if agent.memory.n_entries > agent.batch_size:
            loss = agent.replay()
            if loss > 0:
                losses.append(loss)

        state = next_state
        mask = next_mask
        total_reward += reward

    agent.update_eps()  

    if len(losses) > 0:
        agent.scheduler.step()

    scores.append(total_reward)
    remains.append(env.remain)
    steps.append(step)

    if (episode % 100 == 0): 
        print(f"Episode {episode:6d}  |  Step {np.mean(steps):4.1f}  |  Score {np.mean(scores):6.1f}  |  Loss {np.mean(losses):5.3f}"
            f"  |  Eps {agent.eps:5.2f}  |  Q {np.mean(agent.q_history):5.1f}  |  Rem {np.mean(remains):4.2f}")
        
        history_episodes.append(episode)
        history_remains.append(np.mean(remains))
        history_steps.append(np.mean(steps))
        history_losses.append(np.mean(losses) if len(losses) > 0 else 0)

        if np.mean(steps) > 30.0: 
            if np.mean(remains) < least_rem:
                save_episode = episode
                least_rem = np.mean(remains)
                torch.save(agent.q_network.state_dict(), 'model.pth')
            
print(f"Least Rem {least_rem:.2f}  ( Episode {save_episode} )")

draw(history_episodes, history_remains, history_steps, history_losses)