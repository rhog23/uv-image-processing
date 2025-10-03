import gymnasium as gym

# Buat environment
env = gym.make("MountainCar-v0", render_mode="human")

# Reset awal
observation, info = env.reset(seed=42)

for step in range(500):  # max 500 steps
    action = env.action_space.sample()  # random action
    observation, reward, terminated, truncated, info = env.step(action)

    # Ekstrak state
    position, velocity = observation
    print(f"Step {step}")
    print(f"  Position: {position:.3f}, Velocity: {velocity:.3f}")
    print(f"  Action: {action}, Reward: {reward}\n")

    if terminated or truncated:
        print("Episode selesai!\n")
        observation, info = env.reset()

env.close()
