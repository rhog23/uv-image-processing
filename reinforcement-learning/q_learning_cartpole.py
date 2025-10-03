import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.display import display, clear_output

# ------------------------
# Hyperparameter
# ------------------------
n_episodes = 1000
alpha = 0.1
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

# ------------------------
# Buat environment (training tanpa visualisasi)
# ------------------------
env = gym.make("CartPole-v1")

# Fungsi diskritisasi state
def discretize_state(state, bins):
    indices = []
    for i in range(len(state)):
        idx = np.digitize(state[i], bins[i]) - 1  # mulai dari 0
        idx = min(max(idx, 0), len(bins[i]) - 2)  # clamp index
        indices.append(idx)
    return tuple(indices)

# Buat bins untuk setiap dimensi state
position_bins = np.linspace(-4.8, 4.8, 11)
velocity_bins = np.linspace(-4.0, 4.0, 11)
angle_bins = np.linspace(-0.418, 0.418, 11)
angular_velocity_bins = np.linspace(-2.0, 2.0, 11)
bins = [position_bins, velocity_bins, angle_bins, angular_velocity_bins]

# Hitung jumlah state diskrit
n_bins = [len(b) - 1 for b in bins]
Q = np.zeros(n_bins + [env.action_space.n])  # Q-table multidimensi

# Fungsi pilih action
def choose_action(state_discrete, Q, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    return np.argmax(Q[state_discrete])

# ------------------------
# Training loop
# ------------------------
rewards = []
for episode in range(n_episodes):
    state, _ = env.reset()
    state_discrete = discretize_state(state, bins)
    total_reward = 0
    
    while True:
        action = choose_action(state_discrete, Q, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_state_discrete = discretize_state(next_state, bins)

        # Update Q-value
        best_next_action = np.argmax(Q[next_state_discrete])
        td_target = reward + gamma * Q[next_state_discrete][best_next_action] * (not done)
        td_error = td_target - Q[state_discrete][action]
        Q[state_discrete][action] += alpha * td_error

        state_discrete = next_state_discrete
        total_reward += reward

        if done:
            break
    
    rewards.append(total_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if episode % 100 == 0:
        print(f"Episode {episode}, Average Reward (last 100): {np.mean(rewards[-100:])}")

# ------------------------
# Plot hasil training
# ------------------------
plt.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'))
plt.title('Average Reward over Episodes')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.grid()
plt.show()

# ------------------------
# Testing: Pilih mode visualisasi
# ------------------------
def test_agent(Q, bins, mode="human", steps=300):
    env = gym.make("CartPole-v1", render_mode=mode)
    state, _ = env.reset()
    state_discrete = discretize_state(state, bins)

    for t in range(steps):
        action = np.argmax(Q[state_discrete])
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state_discrete = discretize_state(state, bins)

        if mode == "rgb_array":  # tampilkan frame di notebook
            frame = env.render()
            plt.imshow(frame)
            plt.axis('off')
            display(plt.gcf())
            clear_output(wait=True)
            time.sleep(0.05)

        if done:
            state, _ = env.reset()
            state_discrete = discretize_state(state, bins)

    env.close()

# ------------------------
# Contoh penggunaan:
# ------------------------
# Jika run di lokal python script → gunakan mode="human"
test_agent(Q, bins, mode="human", steps=500)

# Jika run di Jupyter/Colab → gunakan mode="rgb_array"
# test_agent(Q, bins, mode="rgb_array", steps=200)