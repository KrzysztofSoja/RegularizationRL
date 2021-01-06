import time
import numpy as np
import json
import csv
import gym

from typing import Union
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import GymStepReturn


class CyclicMonitor(Monitor):

    def __init__(self, env: gym.Env, max_file_size: int = 100, *args, **kwargs):
        super(CyclicMonitor, self).__init__(*args, env=env, **kwargs)
        self.file_size = 0
        self.max_file_size = max_file_size
        self.env_spec = env.spec
        self.env_spec_id = env.spec.id

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, done, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        if done:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_rewards.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.logger and self.file_size < self.max_file_size:
                self.logger.writerow(ep_info)
                self.file_handler.flush()
                self.file_size += 1
            elif self.logger:
                self.episode_rewards = [ep_rew]
                self.episode_lengths = [ep_len]
                self.episode_times = [time.time() - self.t_start]
                self.file_handler.seek(0)
                self.file_handler.write(
                    "#%s\n" % json.dumps({"t_start": self.t_start, "env_id": self.env_spec and self.env_spec_id}))
                self.logger = csv.DictWriter(self.file_handler,
                                             fieldnames=("r", "l", "t") + self.reset_keywords + self.info_keywords)
                self.logger.writeheader()
                self.logger.writerow(ep_info)
                self.file_handler.flush()
                self.file_handler.truncate()
                self.file_size = 1
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, done, info
