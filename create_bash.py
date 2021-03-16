# ToDo: Dodać iterowanie po liście

ENVIRONMENT_NAMES = ['HalfCheetah-v2'] # , 'Ant-v2', 'Walker2d-v2', 'Hopper-v2', 'Humanoid-v2']
ALGORITHMS = ['PPO', 'A2C', 'SAC', 'DDPG', 'TD3', 'TQC']
STEPS = 1_000
WORKERS = 8
DROPOUT = 0.5  # flout number from 0 to 1 or False
WEIGHT_DECAY = 0.001

lines = []

n_experiment = 0


for environment_name in ENVIRONMENT_NAMES:
    for algorithm in ALGORITHMS:
        command = f"python main.py --env {environment_name} --algo {algorithm} --steps {STEPS} --workers {WORKERS}"

        if not DROPOUT:
            command += f" --dropout {DROPOUT}"

        if not WEIGHT_DECAY:
            command += f" --weight_decay {WEIGHT_DECAY}"

        if n_experiment % 2 == 0:
            lines.append(f"nohup {command} &\n")
        else:
            lines.append(f"{command}\n")
        n_experiment += 1



with open('run_all.sh', 'w') as file:
    file.writelines(lines)
