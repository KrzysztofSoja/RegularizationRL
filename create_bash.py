from itertools import product

from settings import *

from steps import get_steps, is_defined

PATH_TO_MAIN = os.path.abspath(os.path.join(__file__, os.pardir, 'main.py'))


lines = []


n_experiment = 0
for environment_name, algorithm, seed, dropout, weight_decay, entropy_coefficient, manifold_mixup_alpha, \
        gradient_penalty_actor, gradient_penalty_actor_k, gradient_penalty_critic, gradient_penalty_critic_k in \
        product(ENVIRONMENT_NAMES, ALGORITHMS, SEEDS, DROPOUT, WEIGHT_DECAY, ENTROPY_COEFFICIENT, MANIFOLD_MIXUP_ALPHA,
                GRADIENT_PENALTY_ACTOR, GRADIENT_PENALTY_ACTOR_K, GRADIENT_PENALTY_CRITIC, GRADIENT_PENALTY_CRITIC_K):

    if not is_defined(algorithm, environment_name):
        continue

    command = f"{EXECUTABLE} {PATH_TO_MAIN} --env {environment_name} --algo {algorithm} --steps {get_steps(algorithm, environment_name)} --workers {WORKERS}"

    if 'SEEDS' in globals() and seed:
        command += f" --seed {seed}"

    if dropout:
        command += f" --dropout {dropout}"

    if weight_decay:
        command += f" --weight_decay {weight_decay}"

    if entropy_coefficient and algorithm in {'A2C', 'PPO'}:
        command += f" --entropy_coefficient {entropy_coefficient}"

    if manifold_mixup_alpha > 0:
        command += f" --manifold_mixup_alpha {manifold_mixup_alpha}"

    if gradient_penalty_actor > 0:
        command += f" --gradient_penalty_actor {gradient_penalty_actor}"
        command += f" --gradient_penalty_actor_k {gradient_penalty_actor_k}"

    if gradient_penalty_critic > 0:
        command += f" --gradient_penalty_critic {gradient_penalty_critic}"
        command += f" --gradient_penalty_critic_k {gradient_penalty_critic_k}"

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

    if 'TAG' in globals() and TAG:
        command += f' --additional_tag {TAG}'

    lines.append(command + '\n')

with open('run_all.sh', 'w') as file:
    file.writelines(lines)
