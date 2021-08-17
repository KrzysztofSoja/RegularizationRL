# please install this by python venv

pip -V
pip install --upgrade pip

pip install gym

pip install pybullet
git clone https://github.com/benelot/pybullet-gym.git
cd pybullet-gym
pip install -e .
cd ..
rm -rf pybullet-gym

pip install -U 'mujoco-py<2.1,>=2.0'
pip install box2d-py

pip install stable-baselines3==1.0
pip install sb3-contrib==1.0
pip install neptune-client
pip install neptune-contrib
pip install opencv-python
