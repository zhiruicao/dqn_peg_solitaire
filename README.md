# Peg Solitaire with AI Assistant 🧠♟️

A modern, interactive implementation of the classic Peg Solitaire board game, featuring a built-in AI assistant trained via Deep Reinforcement Learning (DRL). 

This project consists of two main components:
1. **The Training Pipeline (Python/PyTorch):** A highly customized Dueling Double Deep Q-Network (D3QN) leveraging Prioritized Experience Replay and Curriculum Learning to master the complex state space of Peg Solitaire.
2. **The Game Engine (C++/Raylib):** A lightweight, interactive frontend that features a completely custom, dependency-free C++ neural network inference engine to run the trained PyTorch model in real-time.

## ✨ Features
* **Playable UI:** A clean, visually appealing 2D interface built with Raylib.
* **Smart AI Assistant:** Press `SPACE` to let the AI calculate and play the optimal next move for you.
* **Custom Inference Engine:** The PyTorch model is exported to a plain JSON file and inferred purely through raw C++ arrays and loops—no heavy ML libraries required in the frontend.

---

## ⚙️ Training Hyperparameters & Reward Formulation

The agent was trained for 100,000 episodes  using a carefully tuned set of hyperparameters and a custom reward function designed to encourage continuous clearing while heavily penalizing stranded pegs.

### **Network**

**1. Feature Extraction Layer**
* **Input**: Reshaped to `(1, 7, 7)`.
* **Convolutional Layer 1**: `32 filters` ($3 \times 3$ kernel) with **ReLU**.
* **Convolutional Layer 2**: `64 filters` ($3 \times 3$ kernel) with **ReLU**.
* **Fully Connected Layer**: `256 units` with **ReLU**.

**2. Dueling Streams**
The 256-unit feature vector is fed into two parallel streams:
* **Value Stream**: `256 units` → **ReLU** → `128 units` → **ReLU** → `1 unit` (Estimates state value $V(s)$).
* **Advantage Stream**: `256 units` → **ReLU** → `128 units` → **ReLU** → `76 units` (Estimates action advantages $A(s, a)$).

**3. Output**
The final Q-values for the **76 possible actions** are calculated by aggregating the Value and Advantage streams.

### **Hyperparameters**
* **Optimizer:** AdamW 
* **Learning Rate Scheduler:** Cosine Annealing (from 5e-5 down to 1e-5)
* **Discount Factor ($\gamma$):** 0.99 
* **Batch Size:** 256 
* **Target Network Update ($\tau$):** 0.005 (Soft update) 
* **Exploration ($\epsilon$):** Decays from 1.0 to 0.01 over the first 80% of episodes 
* **Gradient Clipping:** Max norm of 1.0 
* **PER Capacity:** 100,000 transitions 
* **PER Parameters:** $\alpha = 0.6$, $\beta = 0.4$ (annealing to 1.0 over 90% of training) 

### **Reward Formulation**
* **Standard Move:** $+1.0$ reward per step to encourage the agent to keep moving.
* **Perfect Win:** $+100.0$ if the game ends with exactly 1 peg remaining in the exact center of the board (index 24).
* **Defeat:** $-5.0 \times \text{Remaining Pegs}$ if no valid moves exist.

---

## 🔬 Deep Reinforcement Learning Methodology

Training an agent to solve Peg Solitaire is notoriously difficult due to the massive state space and extremely sparse rewards (a win requires 31 perfect consecutive moves). To tackle this, the agent is built upon four core methodologies:

### 1. Double Dueling DQN
To prevent the catastrophic overestimation of Q-values common in standard DQNs, this project implements Double DQN logic: the online network selects the best action, while the target network evaluates its value. Furthermore, the network utilizes a **Dueling Architecture**, splitting the hidden layers into two separate streams: a Value stream ($V(s)$) and an Advantage stream ($A(s, a)$). These streams are aggregated at the output, allowing the agent to learn which states are inherently valuable regardless of the action taken.

### 2. Convolutional Neural Network
Rather than treating the board as a flat array, the 7x7 board is reshaped into a 1x7x7 2D grid, allowing the network to interpret it as an image. The feature extraction backbone consists of two sequential `Conv2D` layers (32 and 64 channels respectively, with a 3x3 kernel and padding of 1), separated by ReLU activations. This spatial processing is crucial for the agent to recognize local peg formations and valid jumping patterns.

### 3. Prioritized Experience Replay
Instead of sampling past experiences uniformly, the agent learns more effectively by focusing on "surprising" transitions. A custom `SumTree` data structure  is implemented to sample transitions based on their Temporal Difference (TD) error. This ensures that critical mistakes or rare successes (like achieving a perfect win) are replayed and learned from much more frequently than standard, predictable moves.

### 4. Curriculum Learning
Because a random agent will virtually never solve the board by chance, standard RL exploration fails. This project uses a highly effective Curriculum Learning approach that dynamically adjusts the starting difficulty based on the training progression.
* **Phase 1: Reverse Play (0% - 70% of Training):** The environment initializes in the winning state (a single peg in the center) and takes $N$ valid reverse moves to scramble the board. As training progresses, $N$ gradually increases from a few steps to nearly a full board scramble.
* **Phase 2: Forward Play (70% - 100% of Training):** Once the agent learns late-game and mid-game concepts, the environment switches to standard forward play. It begins from the standard starting layout, making a decreasing number of random forward moves to create the starting state for the episode, eventually forcing the agent to master the opening phase of the game.

---

## 🏆 Evaluation

| Initial Pegs | Win Rate | Avg Remain |
| :--- | :--- | :--- |
| 30 | 100.0% | 1.00 |
| 29 | 100.0% | 1.00 |
| 28 | 82.2% | 1.67 | 

---

## 🚀 How to Build and Run

### Running the Game (C++)
The frontend is built using C++ and the Raylib library. The trained model weights are parsed directly from `assets/weights.json`.
1. Ensure you have a C++ compiler and [Raylib](https://www.raylib.com/) installed.
2. Compile the source files (`main.cpp`, `game.cpp`, `engine.cpp`).
3. Run the executable. Click pegs to play, or press `SPACE` to let the AI take over!

### Training the Model (Python)
If you wish to retrain the model or tweak the hyperparameters:
1. Ensure you have `torch`, `numpy`, and `matplotlib` installed.
2. Run the training script: `python train.py`
3. The script will save the training curve as `train_curve.png`  and export the PyTorch weights to a `.pth` file.