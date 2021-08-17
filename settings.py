import os
import sys

# Paths

PATH_TO_RUN = os.path.abspath(os.path.join(__file__, os.pardir, 'run_all.sh'))
PATH_TO_CONTAINER = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, 'mujoco.sif'))
PATH_TO_SBATCH_COLLECTIVE = os.path.abspath(os.path.join(__file__, os.pardir, 'super_run.sh'))


# Environment settings

MUJOCO = ["InvertedDoublePendulum-v2"]
       # ['HalfCheetah-v2',
       #   'Ant-v2',
       #   'Walker2d-v2',
       #   'Hopper-v2',
       #   'Humanoid-v2']

PYBULLET = ['HalfCheetahPyBulletEnv-v0',
            'AntPyBulletEnv-v0',
            'Walker2DPyBulletEnv-v0',
            'HopperPyBulletEnv-v0',
            'HumanoidPyBulletEnv-v0']
ENVIRONMENT_NAMES = MUJOCO


# Algorithms settings

ALGORITHMS = ['PPO', 'A2C', 'SAC', 'TD3']  # ['TQC']
SEEDS = [111, 12345, 777] #, 56765, 97531245]
STEPS = 100_000 #2_500_000
WORKERS = 8


# Algorithms additional parameters
USE_SDE = False


# Regularizers settings

DROPOUT = [False]  # flout number from 0 to 1 or False
WEIGHT_DECAY = [False] #[0.01]  # [0.01, 0.001, 0.0001, 0.00001]
ENTROPY_COEFFICIENT = [False]  # , 0.01, 0.05, 0.1, 0.5, 1]
MANIFOLD_MIXUP_ALPHA = [0]  #[0.1, 0.4, 0.8, 1.0, 2.0, 4.0]
GRADIENT_PENALTY_ACTOR = [0]
GRADIENT_PENALTY_ACTOR_K = [0]
GRADIENT_PENALTY_CRITIC = [0]
GRADIENT_PENALTY_CRITIC_K = [0]


# Logs settings

USE_NEPTUNE = True
MAKE_VIDEO_FREQ = 0 #2_000#1_000_000
VALIDATION_FREQ = 1000
VALIDATION_LENGTH = 1
TAG = 'baseline'
#COMMENT = 'Testing random seed'


# Experiments settings

EXECUTABLE = sys.executable
EXPERIMENT_NAME = "BasicRLConfiguration"
N_NODES = 1
HOURS = "12"
SINGULARITY_COMMAND = F"singularity {PATH_TO_CONTAINER} exec "
