#ML-6
import numpy as np
import random

# Define the Maze
# 0 = free path, 1 = wall
maze = np.array([
    [0, 0, 0, 1, 0],
    [1, 0, 1, 0, 0],
    [0, 0, 0, 0, 1],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0]
])

# Start and Goal positions
start = (0, 0)
goal = (4, 4)

# Define possible actions: up, down, left, right
actions = ['up', 'down', 'left', 'right']

# Initialize Q-table
Q = {}
for i in range(maze.shape[0]):
    for j in range(maze.shape[1]):
        Q[(i, j)] = {a: 0 for a in actions}

# Hyperparameters
alpha = 0.8       # learning rate
gamma = 0.9       # discount factor
epsilon = 0.3     # exploration rate
episodes = 500

# Reward function
def get_reward(state):
    if state == goal:
        return 10
    else:
        return -1  # small penalty to encourage shorter paths

#ML-6
# Function to get next state
def next_state(state, action):
    i, j = state
    if action == 'up':
        i -= 1
    elif action == 'down':
        i += 1
    elif action == 'left':
        j -= 1
    elif action == 'right':
        j += 1
    # Check boundaries and walls
    if i < 0 or i >= maze.shape[0] or j < 0 or j >= maze.shape[1] or maze[i, j] == 1:
        return state  # invalid move → stay in place
    return (i, j)

# Q-learning algorithm
for ep in range(episodes):
    state = start
    done = False
    while not done:
        # ε-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)
        else:
            action = max(Q[state], key=Q[state].get)

        new_state = next_state(state, action)
        reward = get_reward(new_state)

        # Q-update rule
        Q[state][action] = Q[state][action] + alpha * (reward + gamma * max(Q[new_state].values()) - Q[state][action])

        state = new_state
        if state == goal:
            done = True

#ML-6
# Test the learned policy
state = start
path = [state]
while state != goal:
    action = max(Q[state], key=Q[state].get)
    state = next_state(state, action)
    if state in path:  # avoid loops
        break
    path.append(state)

print("🏁 Path found by the agent:")
print(path)
