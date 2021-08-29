__all__ = ['get_steps', 'is_defined']


_steps = {('A2C', 'HalfCheetah-v2'): 2_000_000,
          ('PPO', 'HalfCheetah-v2'): 2_000_000,
          ('TD3', 'HalfCheetah-v2'): 500_000,
          ('SAC', 'HalfCheetah-v2'): 500_000,

          ('A2C', 'Hopper-v2'): 1_500_000,
          ('PPO', 'Hopper-v2'): 1_500_000,
          ('TD3', 'Hopper-v2'): 750_000,
          ('SAC', 'Hopper-v2'): 500_000,

          ('A2C', 'Ant-v2'): 2_000_000,
          ('PPO', 'Ant-v2'): 2_000_000,
          ('TD3', 'Ant-v2'): 1_000_000,
          ('SAC', 'Ant-v2'): 1_000_000,

          ('A2C', 'Walker2d-v2'): 500_000,
          ('PPO', 'Walker2d-v2'): 500_000,
          ('TD3', 'Walker2d-v2'): 500_000,
          ('SAC', 'Walker2d-v2'): 500_000,

          ('SAC', 'Humanoid-v2'): 1_000_000,
          }


def get_steps(algo: str, env: str) -> int:
    return _steps[(algo, env)]


def is_defined(algo: str, env: str) -> bool:
    return (algo, env) in _steps.keys()

