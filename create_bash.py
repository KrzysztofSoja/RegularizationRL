from itertools import product


ENVIRONMENT_NAMES = ['HalfCheetah-v2'] # , 'Ant-v2', 'Walker2d-v2', 'Hopper-v2', 'Humanoid-v2']
ALGORITHMS = ['PPO', 'A2C'] #'SAC', 'DDPG', 'TD3', 'TQC']
STEPS = 1_000
WORKERS = 8
DROPOUT = [False]  # flout number from 0 to 1 or False
WEIGHT_DECAY = [False]
ENTROPY_COEFFICIENT = [False, 0.01, 0.05, 0.1, 0.5, 1]
NOHUP = False

lines = []

n_experiment = 0
for environment_name, algorithm, dropout, weight_decay, entropy_coefficient in \
        product(ENVIRONMENT_NAMES, ALGORITHMS, DROPOUT, WEIGHT_DECAY, ENTROPY_COEFFICIENT):
    command = f"python main.py --env {environment_name} --algo {algorithm} --steps {STEPS} --workers {WORKERS}"

    if dropout:
        command += f" --dropout {dropout}"

    if weight_decay:
        command += f" --weight_decay {weight_decay}"

    if entropy_coefficient and algorithm in {'A2C', 'PPO'}:
        command += f" --entropy_coefficient {entropy_coefficient}"

    if n_experiment % 2 == 0 and NOHUP:
        lines.append(f"nohup {command} &\n")
    else:
        lines.append(f"{command}\n")
    n_experiment += 1


with open('run_all.sh', 'w') as file:
    file.writelines(lines)
