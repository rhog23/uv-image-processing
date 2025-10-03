import gymnasium as gym

# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="human")

# Reset the environment
observation, info = env.reset(seed=42)
print(info)

for step in range(1000):
    action = env.action_space.sample()  # random action
    observation, reward, terminated, truncated, info = env.step(action)

    # Ekstract state
    x, y, vx, vy, theta, theta_dot, left_leg, right_leg = observation

    print(f"Step {step}")
    print(f"  Posisi: x={x:.3f}, y={y:.3f}")
    print(f"  Kecepatan: vx={vx:.3f}, vy={vy:.3f}")
    print(f"  Sudut: {theta:.3f}, Kecepatan Sudut: {theta_dot:.3f}")
    print(f"  Left Leg Contact: {int(left_leg)}, Right Leg Contact: {int(right_leg)}")
    print(f"  Reward: {reward:.3f}\n")

    if terminated or truncated:
        print("Episode selesai!\n")
        observation, info = env.reset()

env.close()
