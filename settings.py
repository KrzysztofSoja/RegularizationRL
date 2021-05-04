import os


# Paths

PATH_TO_RUN = os.path.abspath(os.path.join(__file__, os.pardir, 'run_all.sh'))
PATH_TO_CONTAINER = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, 'mujoco.sif'))
PATH_TO_SBATCH_COLLECTIVE = os.path.abspath(os.path.join(__file__, os.pardir, 'super_run.sh'))


# Environment settings

MUJOCO = ['HalfCheetah-v2',
          'Ant-v2',
          'Walker2d-v2',
          'Hopper-v2',
          'Humanoid-v2']
PYBULLET = ['HalfCheetahPyBulletEnv-v0', ]
           # 'AntPyBulletEnv-v0',
           # 'Walker2DPyBulletEnv-v0',
           # 'HopperPyBulletEnv-v0',
           # 'HumanoidPyBulletEnv-v0']
ENVIRONMENT_NAMES = PYBULLET


# Algorithms settings

ALGORITHMS = ['PPO', 'A2C', 'SAC', 'TQC', 'DDPG', 'TD3']
SEEDS = [111]  #, 12345, 777, 56765, 97531245]
STEPS = 10_000 #2_500_000
WORKERS = 8


# Algorithms additional parameters
USE_SDE = False


# Regularizers settings

DROPOUT = [False]  # flout number from 0 to 1 or False
WEIGHT_DECAY = [False]
ENTROPY_COEFFICIENT = [False]  # , 0.01, 0.05, 0.1, 0.5, 1]


# Logs settings

USE_NEPTUNE = True
MAKE_VIDEO_FREQ = 2_000#1_000_000
VALIDATION_FREQ = 1000
VALIDATION_LENGTH = 1
#COMMENT = "This is testing run with default params for five different random seed."


# Experiments settings

EXECUTABLE = 'python'# sys.executable
EXPERIMENT_NAME = "BasicRLConfiguration"
N_NODES = 1
HOURS = "12"
SINGULARITY_COMMAND = F"singularity exec {PATH_TO_CONTAINER} "

