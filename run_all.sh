#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J BasicRLAlgorithTest
## Liczba alokowanych węzłów
#SBATCH -N 1
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=12:00:00 
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgcontinualrl
## Specyfikacja partycji
#SBATCH -p plgrid
## Plik ze standardowym wyjściem
#SBATCH --output="output.out"
## Plik ze standardowym wyjściem błędów
#SBATCH --error="error.err"


## przejscie do katalogu z ktorego wywolany zostal sbatch
cd $SLURM_SUBMIT_DIR


nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HalfCheetahPyBulletEnv-v0 --algo A2C --steps 2500000 --workers 8 --seed 777 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HalfCheetahPyBulletEnv-v0 --algo A2C --steps 2500000 --workers 8 --seed 56765 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HalfCheetahPyBulletEnv-v0 --algo A2C --steps 2500000 --workers 8 --seed 97531245 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HalfCheetahPyBulletEnv-v0 --algo SAC --steps 2500000 --workers 8 --seed 111 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HalfCheetahPyBulletEnv-v0 --algo SAC --steps 2500000 --workers 8 --seed 12345 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HalfCheetahPyBulletEnv-v0 --algo SAC --steps 2500000 --workers 8 --seed 777 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HalfCheetahPyBulletEnv-v0 --algo SAC --steps 2500000 --workers 8 --seed 56765 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HalfCheetahPyBulletEnv-v0 --algo SAC --steps 2500000 --workers 8 --seed 97531245 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HalfCheetahPyBulletEnv-v0 --algo DDPG --steps 2500000 --workers 8 --seed 111 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HalfCheetahPyBulletEnv-v0 --algo DDPG --steps 2500000 --workers 8 --seed 12345 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HalfCheetahPyBulletEnv-v0 --algo DDPG --steps 2500000 --workers 8 --seed 777 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HalfCheetahPyBulletEnv-v0 --algo DDPG --steps 2500000 --workers 8 --seed 56765 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HalfCheetahPyBulletEnv-v0 --algo DDPG --steps 2500000 --workers 8 --seed 97531245 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HalfCheetahPyBulletEnv-v0 --algo TD3 --steps 2500000 --workers 8 --seed 111 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HalfCheetahPyBulletEnv-v0 --algo TD3 --steps 2500000 --workers 8 --seed 12345 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HalfCheetahPyBulletEnv-v0 --algo TD3 --steps 2500000 --workers 8 --seed 777 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HalfCheetahPyBulletEnv-v0 --algo TD3 --steps 2500000 --workers 8 --seed 56765 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HalfCheetahPyBulletEnv-v0 --algo TD3 --steps 2500000 --workers 8 --seed 97531245 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HalfCheetahPyBulletEnv-v0 --algo TQC --steps 2500000 --workers 8 --seed 111 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HalfCheetahPyBulletEnv-v0 --algo TQC --steps 2500000 --workers 8 --seed 12345 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HalfCheetahPyBulletEnv-v0 --algo TQC --steps 2500000 --workers 8 --seed 777 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HalfCheetahPyBulletEnv-v0 --algo TQC --steps 2500000 --workers 8 --seed 56765 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HalfCheetahPyBulletEnv-v0 --algo TQC --steps 2500000 --workers 8 --seed 97531245 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env AntPyBulletEnv-v0 --algo PPO --steps 2500000 --workers 8 --seed 111 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env AntPyBulletEnv-v0 --algo PPO --steps 2500000 --workers 8 --seed 12345 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env AntPyBulletEnv-v0 --algo PPO --steps 2500000 --workers 8 --seed 777 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env AntPyBulletEnv-v0 --algo PPO --steps 2500000 --workers 8 --seed 56765 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env AntPyBulletEnv-v0 --algo PPO --steps 2500000 --workers 8 --seed 97531245 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env AntPyBulletEnv-v0 --algo A2C --steps 2500000 --workers 8 --seed 111 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env AntPyBulletEnv-v0 --algo A2C --steps 2500000 --workers 8 --seed 12345 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env AntPyBulletEnv-v0 --algo A2C --steps 2500000 --workers 8 --seed 777 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env AntPyBulletEnv-v0 --algo A2C --steps 2500000 --workers 8 --seed 56765 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env AntPyBulletEnv-v0 --algo A2C --steps 2500000 --workers 8 --seed 97531245 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env AntPyBulletEnv-v0 --algo SAC --steps 2500000 --workers 8 --seed 111 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env AntPyBulletEnv-v0 --algo SAC --steps 2500000 --workers 8 --seed 12345 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env AntPyBulletEnv-v0 --algo SAC --steps 2500000 --workers 8 --seed 777 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env AntPyBulletEnv-v0 --algo SAC --steps 2500000 --workers 8 --seed 56765 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env AntPyBulletEnv-v0 --algo SAC --steps 2500000 --workers 8 --seed 97531245 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env AntPyBulletEnv-v0 --algo DDPG --steps 2500000 --workers 8 --seed 111 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env AntPyBulletEnv-v0 --algo DDPG --steps 2500000 --workers 8 --seed 12345 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env AntPyBulletEnv-v0 --algo DDPG --steps 2500000 --workers 8 --seed 777 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env AntPyBulletEnv-v0 --algo DDPG --steps 2500000 --workers 8 --seed 56765 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env AntPyBulletEnv-v0 --algo DDPG --steps 2500000 --workers 8 --seed 97531245 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env AntPyBulletEnv-v0 --algo TD3 --steps 2500000 --workers 8 --seed 111 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env AntPyBulletEnv-v0 --algo TD3 --steps 2500000 --workers 8 --seed 12345 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env AntPyBulletEnv-v0 --algo TD3 --steps 2500000 --workers 8 --seed 777 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env AntPyBulletEnv-v0 --algo TD3 --steps 2500000 --workers 8 --seed 56765 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env AntPyBulletEnv-v0 --algo TD3 --steps 2500000 --workers 8 --seed 97531245 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env AntPyBulletEnv-v0 --algo TQC --steps 2500000 --workers 8 --seed 111 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env AntPyBulletEnv-v0 --algo TQC --steps 2500000 --workers 8 --seed 12345 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env AntPyBulletEnv-v0 --algo TQC --steps 2500000 --workers 8 --seed 777 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env AntPyBulletEnv-v0 --algo TQC --steps 2500000 --workers 8 --seed 56765 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env AntPyBulletEnv-v0 --algo TQC --steps 2500000 --workers 8 --seed 97531245 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env Walker2DPyBulletEnv-v0 --algo PPO --steps 2500000 --workers 8 --seed 111 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env Walker2DPyBulletEnv-v0 --algo PPO --steps 2500000 --workers 8 --seed 12345 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env Walker2DPyBulletEnv-v0 --algo PPO --steps 2500000 --workers 8 --seed 777 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env Walker2DPyBulletEnv-v0 --algo PPO --steps 2500000 --workers 8 --seed 56765 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env Walker2DPyBulletEnv-v0 --algo PPO --steps 2500000 --workers 8 --seed 97531245 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env Walker2DPyBulletEnv-v0 --algo A2C --steps 2500000 --workers 8 --seed 111 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env Walker2DPyBulletEnv-v0 --algo A2C --steps 2500000 --workers 8 --seed 12345 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env Walker2DPyBulletEnv-v0 --algo A2C --steps 2500000 --workers 8 --seed 777 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env Walker2DPyBulletEnv-v0 --algo A2C --steps 2500000 --workers 8 --seed 56765 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env Walker2DPyBulletEnv-v0 --algo A2C --steps 2500000 --workers 8 --seed 97531245 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env Walker2DPyBulletEnv-v0 --algo SAC --steps 2500000 --workers 8 --seed 111 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env Walker2DPyBulletEnv-v0 --algo SAC --steps 2500000 --workers 8 --seed 12345 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env Walker2DPyBulletEnv-v0 --algo SAC --steps 2500000 --workers 8 --seed 777 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env Walker2DPyBulletEnv-v0 --algo SAC --steps 2500000 --workers 8 --seed 56765 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env Walker2DPyBulletEnv-v0 --algo SAC --steps 2500000 --workers 8 --seed 97531245 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env Walker2DPyBulletEnv-v0 --algo DDPG --steps 2500000 --workers 8 --seed 111 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env Walker2DPyBulletEnv-v0 --algo DDPG --steps 2500000 --workers 8 --seed 12345 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env Walker2DPyBulletEnv-v0 --algo DDPG --steps 2500000 --workers 8 --seed 777 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env Walker2DPyBulletEnv-v0 --algo DDPG --steps 2500000 --workers 8 --seed 56765 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env Walker2DPyBulletEnv-v0 --algo DDPG --steps 2500000 --workers 8 --seed 97531245 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env Walker2DPyBulletEnv-v0 --algo TD3 --steps 2500000 --workers 8 --seed 111 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env Walker2DPyBulletEnv-v0 --algo TD3 --steps 2500000 --workers 8 --seed 12345 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env Walker2DPyBulletEnv-v0 --algo TD3 --steps 2500000 --workers 8 --seed 777 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env Walker2DPyBulletEnv-v0 --algo TD3 --steps 2500000 --workers 8 --seed 56765 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env Walker2DPyBulletEnv-v0 --algo TD3 --steps 2500000 --workers 8 --seed 97531245 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env Walker2DPyBulletEnv-v0 --algo TQC --steps 2500000 --workers 8 --seed 111 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env Walker2DPyBulletEnv-v0 --algo TQC --steps 2500000 --workers 8 --seed 12345 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env Walker2DPyBulletEnv-v0 --algo TQC --steps 2500000 --workers 8 --seed 777 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env Walker2DPyBulletEnv-v0 --algo TQC --steps 2500000 --workers 8 --seed 56765 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env Walker2DPyBulletEnv-v0 --algo TQC --steps 2500000 --workers 8 --seed 97531245 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HopperPyBulletEnv-v0 --algo PPO --steps 2500000 --workers 8 --seed 111 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HopperPyBulletEnv-v0 --algo PPO --steps 2500000 --workers 8 --seed 12345 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HopperPyBulletEnv-v0 --algo PPO --steps 2500000 --workers 8 --seed 777 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HopperPyBulletEnv-v0 --algo PPO --steps 2500000 --workers 8 --seed 56765 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HopperPyBulletEnv-v0 --algo PPO --steps 2500000 --workers 8 --seed 97531245 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HopperPyBulletEnv-v0 --algo A2C --steps 2500000 --workers 8 --seed 111 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HopperPyBulletEnv-v0 --algo A2C --steps 2500000 --workers 8 --seed 12345 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HopperPyBulletEnv-v0 --algo A2C --steps 2500000 --workers 8 --seed 777 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HopperPyBulletEnv-v0 --algo A2C --steps 2500000 --workers 8 --seed 56765 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HopperPyBulletEnv-v0 --algo A2C --steps 2500000 --workers 8 --seed 97531245 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HopperPyBulletEnv-v0 --algo SAC --steps 2500000 --workers 8 --seed 111 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HopperPyBulletEnv-v0 --algo SAC --steps 2500000 --workers 8 --seed 12345 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HopperPyBulletEnv-v0 --algo SAC --steps 2500000 --workers 8 --seed 777 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HopperPyBulletEnv-v0 --algo SAC --steps 2500000 --workers 8 --seed 56765 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HopperPyBulletEnv-v0 --algo SAC --steps 2500000 --workers 8 --seed 97531245 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HopperPyBulletEnv-v0 --algo DDPG --steps 2500000 --workers 8 --seed 111 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HopperPyBulletEnv-v0 --algo DDPG --steps 2500000 --workers 8 --seed 12345 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HopperPyBulletEnv-v0 --algo DDPG --steps 2500000 --workers 8 --seed 777 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HopperPyBulletEnv-v0 --algo DDPG --steps 2500000 --workers 8 --seed 56765 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HopperPyBulletEnv-v0 --algo DDPG --steps 2500000 --workers 8 --seed 97531245 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HopperPyBulletEnv-v0 --algo TD3 --steps 2500000 --workers 8 --seed 111 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HopperPyBulletEnv-v0 --algo TD3 --steps 2500000 --workers 8 --seed 12345 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HopperPyBulletEnv-v0 --algo TD3 --steps 2500000 --workers 8 --seed 777 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HopperPyBulletEnv-v0 --algo TD3 --steps 2500000 --workers 8 --seed 56765 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HopperPyBulletEnv-v0 --algo TD3 --steps 2500000 --workers 8 --seed 97531245 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HopperPyBulletEnv-v0 --algo TQC --steps 2500000 --workers 8 --seed 111 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HopperPyBulletEnv-v0 --algo TQC --steps 2500000 --workers 8 --seed 12345 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HopperPyBulletEnv-v0 --algo TQC --steps 2500000 --workers 8 --seed 777 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HopperPyBulletEnv-v0 --algo TQC --steps 2500000 --workers 8 --seed 56765 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HopperPyBulletEnv-v0 --algo TQC --steps 2500000 --workers 8 --seed 97531245 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HumanoidPyBulletEnv-v0 --algo PPO --steps 2500000 --workers 8 --seed 111 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HumanoidPyBulletEnv-v0 --algo PPO --steps 2500000 --workers 8 --seed 12345 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HumanoidPyBulletEnv-v0 --algo PPO --steps 2500000 --workers 8 --seed 777 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HumanoidPyBulletEnv-v0 --algo PPO --steps 2500000 --workers 8 --seed 56765 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HumanoidPyBulletEnv-v0 --algo PPO --steps 2500000 --workers 8 --seed 97531245 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HumanoidPyBulletEnv-v0 --algo A2C --steps 2500000 --workers 8 --seed 111 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HumanoidPyBulletEnv-v0 --algo A2C --steps 2500000 --workers 8 --seed 12345 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HumanoidPyBulletEnv-v0 --algo A2C --steps 2500000 --workers 8 --seed 777 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HumanoidPyBulletEnv-v0 --algo A2C --steps 2500000 --workers 8 --seed 56765 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HumanoidPyBulletEnv-v0 --algo A2C --steps 2500000 --workers 8 --seed 97531245 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HumanoidPyBulletEnv-v0 --algo SAC --steps 2500000 --workers 8 --seed 111 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HumanoidPyBulletEnv-v0 --algo SAC --steps 2500000 --workers 8 --seed 12345 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HumanoidPyBulletEnv-v0 --algo SAC --steps 2500000 --workers 8 --seed 777 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HumanoidPyBulletEnv-v0 --algo SAC --steps 2500000 --workers 8 --seed 56765 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HumanoidPyBulletEnv-v0 --algo SAC --steps 2500000 --workers 8 --seed 97531245 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HumanoidPyBulletEnv-v0 --algo DDPG --steps 2500000 --workers 8 --seed 111 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HumanoidPyBulletEnv-v0 --algo DDPG --steps 2500000 --workers 8 --seed 12345 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HumanoidPyBulletEnv-v0 --algo DDPG --steps 2500000 --workers 8 --seed 777 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HumanoidPyBulletEnv-v0 --algo DDPG --steps 2500000 --workers 8 --seed 56765 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HumanoidPyBulletEnv-v0 --algo DDPG --steps 2500000 --workers 8 --seed 97531245 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HumanoidPyBulletEnv-v0 --algo TD3 --steps 2500000 --workers 8 --seed 111 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HumanoidPyBulletEnv-v0 --algo TD3 --steps 2500000 --workers 8 --seed 12345 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HumanoidPyBulletEnv-v0 --algo TD3 --steps 2500000 --workers 8 --seed 777 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HumanoidPyBulletEnv-v0 --algo TD3 --steps 2500000 --workers 8 --seed 56765 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HumanoidPyBulletEnv-v0 --algo TD3 --steps 2500000 --workers 8 --seed 97531245 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HumanoidPyBulletEnv-v0 --algo TQC --steps 2500000 --workers 8 --seed 111 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HumanoidPyBulletEnv-v0 --algo TQC --steps 2500000 --workers 8 --seed 12345 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HumanoidPyBulletEnv-v0 --algo TQC --steps 2500000 --workers 8 --seed 777 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HumanoidPyBulletEnv-v0 --algo TQC --steps 2500000 --workers 8 --seed 56765 &
nohup srun /net/people/plgkrzysztofsoj/venv/bin/python /net/people/plgkrzysztofsoj/MujocoRunner/main.py --env HumanoidPyBulletEnv-v0 --algo TQC --steps 2500000 --workers 8 --seed 97531245 &
