ENVIRONMENT_NAMES = ['HalfCheetah-v2', 'Ant-v2', 'Walker2d-v2', 'Walker2d-v2', 'Hopper-v2', 'Humanoid-v2']
ALGORITHMS = ['PPO', 'A2C', 'SAC', 'DQN', 'DDPG', 'TD3']
STEPS = 250_000
WORKERS = 8

lines = []

for environment_name in ENVIRONMENT_NAMES:
    for algorithm in ALGORITHMS:
        lines.append(f"python main.py --env {environment_name} --algo {algorithm} --steps {STEPS} --workers {WORKERS}\n")

with open('run_all.sh', 'w') as file:
    file.writelines(lines)
