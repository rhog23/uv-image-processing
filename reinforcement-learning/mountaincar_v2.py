import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

# ------------------------
# Hyperparameter
# ------------------------
n_episodes = 5000       # butuh banyak episode, MountainCar lebih sulit dari CartPole
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.999   # pelan-pelan turunkan epsilon

# ------------------------
# Buat environment
# ------------------------
env = gym.make("MountainCar-v0")

# State space: [position, velocity]
pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 40)
vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 40)

def discretize_state(state):
    pos, vel = state
    pos_idx = np.digitize(pos, pos_space) - 1
    vel_idx = np.digitize(vel, vel_space) - 1
    pos_idx = np.clip(pos_idx, 0, len(pos_space)-1)
    vel_idx = np.clip(vel_idx, 0, len(vel_space)-1)
    return (pos_idx, vel_idx)

# Q-table: (pos_bins, vel_bins, action_space)
Q = np.zeros((len(pos_space), len(vel_space), env.action_space.n))

def choose_action(state_disc, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    return np.argmax(Q[state_disc])

# ------------------------
# Training
# ------------------------
rewards = []
for episode in range(n_episodes):
    state, _ = env.reset()
    state_disc = discretize_state(state)
    total_reward = 0
    
    done = False
    while not done:
        action = choose_action(state_disc, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state_disc = discretize_state(next_state)
        
        # Q-learning update
        best_next_action = np.argmax(Q[next_state_disc])
        td_target = reward + gamma * Q[next_state_disc][best_next_action] * (not done)
        td_error = td_target - Q[state_disc][action]
        Q[state_disc][action] += alpha * td_error
        
        state_disc = next_state_disc
        total_reward += reward
    
    rewards.append(total_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    if episode % 500 == 0:
        print(f"Episode {episode}, Average Reward (last 100): {np.mean(rewards[-100:])}")

# ------------------------
# Plot hasil training
# ------------------------
plt.plot(np.convolve(rewards, np.ones(100)/100, mode="valid"))
plt.title("MountainCar - Average Reward")
plt.xlabel("Episode")
plt.ylabel("Average Reward (100-episode mean)")
plt.grid()
plt.show()

# ------------------------
# Testing agent terlatih dengan visualisasi
# ------------------------
env = gym.make("MountainCar-v0", render_mode="human")
state, _ = env.reset()
state_disc = discretize_state(state)

for t in range(500):  # max steps
    action = np.argmax(Q[state_disc])  # greedy action (tanpa epsilon)
    state, reward, terminated, truncated, _ = env.step(action)
    state_disc = discretize_state(state)
    time.sleep(0.02)  # supaya animasi tidak terlalu cepat
    
    if terminated or truncated:
        state, _ = env.reset()
        state_disc = discretize_state(state)

env.close()
