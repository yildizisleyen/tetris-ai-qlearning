# Game Artificial Intelligence for Tetris: Q-Learning vs. Greedy Heuristic

An experimental Artificial Intelligence study developed as part of graduate AI coursework, comparing a **Reinforcement Learning (Q-Learning)** agent against a **Greedy Heuristic Search** agent in the dynamic environment of **Tetris**.

---

## Overview & Methodology

Tetris features simple rules (`10x20` grid and 7 standard tetromino shapes: `I, O, T, L, J, S, Z`) but a large combinatorial state-action space (up to 40 placement/rotation actions per piece). This project implements and evaluates two distinct decision-making paradigms:

### 1. Greedy Heuristic Agent (`Code/GreedyAlgorithm.py`)
Simulates all valid rotations and horizontal placements for the active tetromino at each step and selects the immediate best move using a weighted multi-factor heuristic evaluation function:
$$\text{Score} = 1.0 \times (\text{Lines Cleared}) - 0.5 \times (\text{Max Height}) - 0.7 \times (\text{Holes}) - 0.3 \times (\text{Bumpiness})$$
* Includes a real-time **Pygame** visual simulation displaying cleared lines, step duration, and total score.

### 2. Reinforcement Learning Agent (`Code/QLearning.py`)
Learns an action-value policy via **Tabular Q-Learning** with an $\epsilon$-greedy exploration strategy ($\epsilon = 0.1$, learning rate $\alpha = 0.1$, discount factor $\gamma = 0.9$):
$$Q(s, a) \leftarrow (1 - \alpha)\,Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') \right]$$
* **Reward Structure:** $+10$ per cleared line, $-1$ step penalty, and $-100$ terminal penalty on game over.
* Includes a live **Matplotlib** board visualization tracking episode progression and cumulative rewards.

---

## Evaluation Metrics

Both agents are evaluated across four core metrics:
1. **Cleared Lines:** Total horizontal rows completed before termination.
2. **Total Score:** Cumulative reward / lines cleared across steps.
3. **Episode Duration:** Survival time / steps per episode.
4. **Average Score over Episodes:** Learning stability and long-term convergence across episodes.

---

## Project Structure

```text
├── Code/
│   ├── GreedyAlgorithm.py          # Greedy heuristic agent with Pygame visualization
│   └── QLearning.py                # Tabular Q-Learning agent with Matplotlib visualization
├── AI_FinalReport_YildizIsleyen.pdf # Full academic report (Literature Review, Methodology, Error Analysis)
└── requirements.txt                # Python dependencies (numpy, pygame, matplotlib)
```

---

## Installation & Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the **Greedy Heuristic Agent** (Pygame window):
```bash
python Code/GreedyAlgorithm.py
```

3. Run the **Q-Learning Agent** (Matplotlib interactive plot):
```bash
python Code/QLearning.py
```
