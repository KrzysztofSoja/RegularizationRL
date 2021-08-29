from itertools import product

from settings import *

from steps import get_steps, is_defined

PATH_TO_MAIN = os.path.abspath(os.path.join(__file__, os.pardir, 'main.py'))


def change_seed(algo, env, seed):
    CHANGING_LIST = {('TD3', 'Hopper-v2', 12345): 24690,
                     ('DDPG', 'Hopper-v2', 12345): 24690,
                     ('TD3', 'Walker2d-v2', 56765): 113530,
                     ('DDPG', 'Hopper-v2', 111): 222,
                     ('DDPG', 'Walker2d-v2', 111): 222}
    if (algo, env, seed) in CHANGING_LIST.keys():
        return CHANGING_LIST[(algo, env, seed)]
    else seed               


lines = []


n_experiment = 0
for environment_name, algorithm, seed, dropout_rate_critic, dropout_rate_actor, dropout_only_on_last_layer, weight_decay,\
    entropy_coefficient, manifold_mixup_alpha, gradient_penalty_actor, gradient_penalty_actor_k, gradient_penalty_critic, \
    gradient_penalty_critic_k, workers in \
        product(ENVIRONMENT_NAMES, ALGORITHMS, SEEDS, DROPOUT_RATE_ACTOR, DROPOUT_RATE_CRITIC,
                DROPOUT_ONLY_ON_LAST_LAYER, WEIGHT_DECAY, ENTROPY_COEFFICIENT, MANIFOLD_MIXUP_ALPHA,
                GRADIENT_PENALTY_ACTOR, GRADIENT_PENALTY_ACTOR_K, GRADIENT_PENALTY_CRITIC, GRADIENT_PENALTY_CRITIC_K, WORKERS):

    #if not is_defined(algorithm, environment_name):
    #    continue

    command = f"{EXECUTABLE} {PATH_TO_MAIN} --env {environment_name} --algo {algorithm} --steps {STEPS}"
    
    if algorithm == 'A2C':
        command += f" --workers {8}"
    elif algorithm == 'PPO':
        command += f" --workers {1}"

    if 'SEEDS' in globals() and seed:
        command += f" --seed {seed}"

    if dropout_rate_critic:
        command += f" --dropout_rate_critic {dropout_rate_critic}"

    if dropout_rate_actor:
        command += f" --dropout_rate_critic {dropout_rate_actor}"

    # Default value is True
    if dropout_rate_critic > 0 or dropout_rate_actor > 0:
        command += f" --dropout_only_on_last_layer {dropout_only_on_last_layer}"

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
