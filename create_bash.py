from itertools import product


ENVIRONMENT_NAMES = ['HalfCheetah-v2'] # , 'Ant-v2', 'Walker2d-v2', 'Hopper-v2', 'Humanoid-v2']
ALGORITHMS = ['PPO', 'A2C', 'SAC', 'DDPG', 'TD3', 'TQC']
STEPS = 1_000
WORKERS = 8
DROPOUT = [False, 0.2, 0.5]  # flout number from 0 to 1 or False
WEIGHT_DECAY = [False, 0.001, 0.005, 0.01]
NOHUP = False

lines = []

n_experiment = 0
for environment_name, algorithm, dropout, weight_decay in product(ENVIRONMENT_NAMES, ALGORITHMS, DROPOUT, WEIGHT_DECAY):
    command = f"python main.py --env {environment_name} --algo {algorithm} --steps {STEPS} --workers {WORKERS}"

    if dropout:
        command += f" --dropout {dropout}"

    if weight_decay:
        command += f" --weight_decay {weight_decay}"

    if n_experiment % 2 == 0 and NOHUP:
        lines.append(f"nohup {command} &\n")
    else:
        lines.append(f"{command}\n")
    n_experiment += 1



with open('run_all.sh', 'w') as file:
    file.writelines(lines)
