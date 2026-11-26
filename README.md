# Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium
## Name: P SOWMIYA
## Date: 26/11/26
## Reg no:212225240152
## Aim

To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

---

## Problem Statement
To train an agent in the FrozenLake environment using Q-Learning to reach the goal while avoiding holes and learn the best possible policy.

## Software Requirements
Python 3.x
Jupyter Notebook / Google Colab
Gymnasium
NumPy
Matplotlib (optional)


## Environment Description

Grid: 8 × 8
States: 64
Q-table: 64 × 4
Value function: reshape(8, 8)
Policy: 64 actions.

## Theory

Q-Learning estimates the optimal action-value function directly.

The action-value function $Q(s,a)$ represents the expected return obtained when the agent takes action $a$ in state $s$, and then follows the best possible policy afterward.

The Q-Learning update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |
| $max_{a} Q(S_{t+1},a)$ | Maximum action value in the next state |

---

## Epsilon-Greedy Action Selection

During training, the agent uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_{a} Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---

## Algorithm

Initialize the Q-table with zeros.
State learning rate, discount factor, and epsilon.
Observe the current state.
Select an action using the epsilon-greedy policy.
Perform the action and observe the reward and next state.
Update the Q-value using the Q-Learning formula.
Repeat until the episode ends.
Train for multiple episodes and extract the learned policy.

## Python Program
```
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
# -------------------------------------------------
# Create FrozenLake Environment
env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------
num_episodes = 100000
alpha = 0.1
gamma = 0.99

epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.99995

max_steps = 200
# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------
q_table = np.zeros((env.observation_space.n,
                    env.action_space.n))


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------
if np.random.random() < epsilon:
    action = env.action_space.sample()
else:
    action = np.argmax(q_table[state])
for episode in range(episodes):
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, info = env.step(action)

        q_table[state, action] += alpha * (
            reward + gamma * np.max(q_table[next_state])
            - q_table[state, action]
        )

        state = next_state
        total_reward += reward
        done = terminated or truncated

    rewards.append(total_reward)

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
# -------------------------------------------------
# Display Functions
# -------------------------------------------------

def print_value_function(values):
    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))


def print_policy(policy):
    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("\nLearned Policy:")
    print(policy_grid)
# Final Q-table
print("\nFinal Q-table:")
print(np.round(q_table, 3))

# Estimated State-Value Function
state_values = np.max(q_table, axis=1)
print("\nEstimated State-Value Function:")
print(np.round(state_values, 3))

# Learned Policy
learned_policy = np.argmax(q_table, axis=1)
print("\nLearned Policy:")
print(learned_policy)
window = 500

moving_average = np.convolve(
    rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))
plt.plot(moving_average)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Q-Learning Curve - FrozenLake")
plt.grid(True)
plt.show()

env.close()
```


## Output
<img width="417" height="645" alt="Screenshot 2026-11-26 120335" src="https://github.com/user-attachments/assets/8c9b4df0-6288-47c4-bd37-d839094b9add" />

<img width="732" height="642" alt="Screenshot 2026-11-26 120341" src="https://github.com/user-attachments/assets/cd651aa2-3406-4b05-86a8-3a84745775a4" />

<img width="892" height="623" alt="Screenshot 2026-11-26 120326" src="https://github.com/user-attachments/assets/d5e355da-1679-493d-8864-a094af6951d8" />



## Result
The Q-Learning control algorithm was successfully implemented using Gymnasium FrozenLake-v1. The agent learned action values and developed a policy to reach the goal while avoiding holes.

## InferenceQ-Learning is an off-policy reinforcement learning algorithm.

It learns the optimal policy using rewards and maximum future Q-values.
Epsilon-greedy helps balance exploration and exploitation.
The agent improves its performance through repeated training.


---

