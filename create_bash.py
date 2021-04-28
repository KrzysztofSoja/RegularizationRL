import sys

from itertools import product


MUJOCO = ['HalfCheetah-v2']  # , 'Ant-v2', 'Walker2d-v2', 'Hopper-v2', 'Humanoid-v2']
PYBULLET = ['HalfCheetahPyBulletEnv-v0', 'AntPyBulletEnv-v0', 'Walker2DPyBulletEnv-v0', 'HopperPyBulletEnv-v0', 'HumanoidPyBulletEnv-v0']

ENVIRONMENT_NAMES = PYBULLET
ALGORITHMS = ['PPO', 'A2C', 'SAC', 'DDPG', 'TD3', 'TQC']
SEEDS = [111, 12345, 777, 56765, 97531245]
STEPS = 2_500_000
WORKERS = 8
DROPOUT = [False]  # flout number from 0 to 1 or False
WEIGHT_DECAY = [False]
ENTROPY_COEFFICIENT = [False] # , 0.01, 0.05, 0.1, 0.5, 1]

SLURM = False
NOHUP = False

lines = []

n_experiment = 0
for environment_name, algorithm, seed, dropout, weight_decay, entropy_coefficient in \
        product(ENVIRONMENT_NAMES, ALGORITHMS, SEEDS, DROPOUT, WEIGHT_DECAY, ENTROPY_COEFFICIENT):
    command = f"{sys.executable} main.py --env {environment_name} --algo {algorithm} --steps {STEPS} --workers {WORKERS}"

    if seed:
        command += f" --dropout {seed}"

    if dropout:
        command += f" --dropout {dropout}"

    if weight_decay:
        command += f" --weight_decay {weight_decay}"

    if entropy_coefficient and algorithm in {'A2C', 'PPO'}:
        command += f" --entropy_coefficient {entropy_coefficient}"

    if SLURM:
        command = 'srun ' + command

    if n_experiment % 2 == 0 and NOHUP:
        lines.append(f"nohup {command} &\n")
    else:
        lines.append(f"{command}\n")
    n_experiment += 1


with open('run_all.sh', 'w') as file:
    file.writelines(lines)
