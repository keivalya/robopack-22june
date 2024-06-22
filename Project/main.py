import numpy as np
import random
import tkinter as tk

# Define the maze environment
maze = [
    [0, 0, 0, 0, 0, 1],
    [0, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0],
    [1, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0]
]

# Define parameters
start = (0, 0)
goal = (5, 5)
actions = ['up', 'down', 'left', 'right']
reverse_actions = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
epsilon = 1.0  # Initial exploration rate
min_epsilon = 0.1
epsilon_decay = 0.995
alpha = 0.1    # Learning rate
gamma = 0.9    # Discount factor

# Initialize Q-table
q_table = np.zeros((len(maze), len(maze[0]), len(actions)))

# Function to get the next state given the current state and action
def get_next_state(state, action):
    i, j = state
    if action == 'up':
        i = max(i - 1, 0)
    elif action == 'down':
        i = min(i + 1, len(maze) - 1)
    elif action == 'left':
        j = max(j - 1, 0)
    elif action == 'right':
        j = min(j + 1, len(maze[0]) - 1)
    return (i, j)

# Function to get the reward for a given state
def get_reward(state):
    if state == goal:
        return 100  # Reward for reaching the goal
    elif maze[state[0]][state[1]] == 1:
        return -100  # Penalty for hitting a wall
    else:
        return -1  # Small penalty for each step

# Function to choose the next action based on epsilon-greedy policy
def choose_action(state, previous_action=None):
    valid_actions = actions[:]
    if previous_action:
        valid_actions.remove(reverse_actions[previous_action])

    if random.uniform(0, 1) < epsilon:
        return random.choice(valid_actions)
    else:
        state_q_values = q_table[state[0], state[1], :]
        state_q_values = [state_q_values[actions.index(action)] if action in valid_actions else -np.inf for action in actions]
        return actions[np.argmax(state_q_values)]

# Train the robot and generate multiple paths
num_episodes = 5000
num_paths = 15
unique_paths = set()
all_paths = []

for path_index in range(num_paths):
    epsilon = 1.0  # Reset epsilon for each path
    for episode in range(num_episodes):
        state = start
        previous_action = None
        while state != goal:
            action = choose_action(state, previous_action)
            next_state = get_next_state(state, action)
            reward = get_reward(next_state)

            # Update Q-value
            current_q = q_table[state[0], state[1], actions.index(action)]
            max_future_q = np.max(q_table[next_state[0], next_state[1], :])
            new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)
            q_table[state[0], state[1], actions.index(action)] = new_q

            state = next_state
            previous_action = action

        # Decay epsilon
        if epsilon > min_epsilon:
            epsilon *= epsilon_decay

    # Generate path after training
    state = start
    path = [state]
    previous_action = None
    while state != goal:
        action = choose_action(state, previous_action)
        next_state = get_next_state(state, action)
        if maze[next_state[0]][next_state[1]] == 1:
            break  # Hit a wall, stop the current path
        state = next_state
        path.append(state)
        previous_action = action
    
    # Add unique paths to the list
    path_tuple = tuple(path)
    if state == goal and path_tuple not in unique_paths:
        unique_paths.add(path_tuple)
        all_paths.append(path)

# Determine the most optimized path
optimized_path = min(all_paths, key=len)

# Tkinter visualization
cell_size = 50

def draw_maze(canvas):
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            color = 'white'
            if maze[i][j] == 1:
                color = 'black'
            elif (i, j) == start:
                color = 'green'
            elif (i, j) == goal:
                color = 'red'
            canvas.create_rectangle(j * cell_size, i * cell_size,
                                    (j + 1) * cell_size, (i + 1) * cell_size,
                                    fill=color)

def draw_path(canvas, path, color='blue'):
    for (i, j) in path:
        canvas.create_oval(j * cell_size + 10, i * cell_size + 10,
                           (j + 1) * cell_size - 10, (i + 1) * cell_size - 10,
                           fill=color)

root = tk.Tk()
root.title("Q-learning Maze Navigation")
canvas = tk.Canvas(root, width=cell_size * len(maze[0]), height=cell_size * len(maze))
canvas.pack()

draw_maze(canvas)

def animate_paths(index=0):
    if index < len(all_paths):
        path = all_paths[index]
        for step in path:
            draw_maze(canvas)
            draw_path(canvas, path[:path.index(step) + 1])
            root.update()
            canvas.after(300)  # Pause for 300 milliseconds
        print(f"Path {index + 1}: {path}")
        total_reward = sum(get_reward(state) for state in path)
        print(f"Total reward for path {index + 1}: {total_reward}")
        root.after(1000, lambda: animate_paths(index + 1))  # Pause for 1 second before next path
    else:
        draw_maze(canvas)
        draw_path(canvas, optimized_path, color='red')
        print("Most optimized path:", optimized_path)
        total_reward = sum(get_reward(state) for state in optimized_path)
        print(f"Total reward for the most optimized path: {total_reward}")

root.after(1000, animate_paths)  # Start showing paths after 1 second
root.mainloop()
