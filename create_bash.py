from itertools import product

from settings import *


PATH_TO_MAIN = os.path.abspath(os.path.join(__file__, os.pardir, 'main.py'))


lines = []

n_experiment = 0
for environment_name, algorithm, seed, dropout, weight_decay, entropy_coefficient in \
        product(ENVIRONMENT_NAMES, ALGORITHMS, SEEDS, DROPOUT, WEIGHT_DECAY, ENTROPY_COEFFICIENT):
    command = f"{EXECUTABLE} {PATH_TO_MAIN} --env {environment_name} --algo {algorithm} --steps {STEPS} --workers {WORKERS}"

    if 'SEEDS' in globals() and seed:
        command += f" --seed {seed}"

    if dropout:
        command += f" --dropout {dropout}"

    if weight_decay:
        command += f" --weight_decay {weight_decay}"

    if entropy_coefficient and algorithm in {'A2C', 'PPO'}:
        command += f" --entropy_coefficient {entropy_coefficient}"

    if algorithm in {'PPO', 'A2C', 'SAC', 'TQC'} and USE_SDE:
        command += ' --use_sde'

    if USE_NEPTUNE:
        command += ' --use_neptune'

    if 'COMMENT' in globals() and COMMENT:
        command += f' --comment "{COMMENT}"'

    if 'MAKE_VIDEO_FREQ' in globals() and MAKE_VIDEO_FREQ:
        command += f' --make_video_freq {MAKE_VIDEO_FREQ}'

    if 'VALIDATION_FREQ' in globals() and VALIDATION_FREQ:
        command += f' --validation_freq {VALIDATION_FREQ}'

    if 'VALIDATION_LENGTH' in globals() and VALIDATION_LENGTH:
        command += f' --validation_length {VALIDATION_LENGTH}'

    lines.append(command + '\n')

with open('run_all.sh', 'w') as file:
    file.writelines(lines)
