import csv
import os
import time
from typing import List, Optional, Tuple, Union

import gym
import numpy as np

from stable_baselines3.common.type_aliases import GymObs, GymStepReturn


class CyclicMonitor(gym.Wrapper):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.

    :param env: The environment
    :param filename: the location to save a log file, can be None for no log
    :param allow_early_resets: allows the reset of the environment before it is done
    :param reset_keywords: extra keywords for the reset call,
        if extra parameters are needed at reset
    :param info_keywords: extra information to log, from the information return of env.step()
    """

    EXT = "monitor.csv"

    def __init__(
        self,
        env: gym.Env,
        filename: Optional[str] = None,
        allow_early_resets: bool = True,
        reset_keywords: Tuple[str, ...] = (),
        info_keywords: Tuple[str, ...] = (),
        max_file_size: int = 100
    ):
        super(CyclicMonitor, self).__init__(env=env)
        self.t_start = time.time()
        if filename is None:
            self.file_handler = None
            self.logger = None
        else:
            if not filename.endswith(CyclicMonitor.EXT):
                if os.path.isdir(filename):
                    filename = os.path.join(filename, CyclicMonitor.EXT)
                else:
                    filename = filename + "." + CyclicMonitor.EXT
            self.file_handler = open(filename, "wt")
            self.logger = csv.DictWriter(self.file_handler, fieldnames=("r", "l", "t") + reset_keywords + info_keywords)
            self.logger.writeheader()
            self.file_handler.flush()

        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.current_reset_info = {}  # extra info about the current episode, that was passed in during reset()

        self.file_size = 0
        self.max_file_size = max_file_size
        self.env_spec = env.spec
        self.env_spec_id = env.spec.id

    def reset(self, **kwargs) -> GymObs:
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError(
                "Tried to reset an environment before done. If you want to allow early resets, "
                "wrap your env with Monitor(env, path, allow_early_resets=True)"
            )
        self.rewards = []
        self.t_start = 0
        self.needs_reset = False
        for key in self.reset_keywords:
            value = kwargs.get(key)
            if value is None:
                raise ValueError("Expected you to pass kwarg {} into reset".format(key))
            self.current_reset_info[key] = value
        return self.env.reset(**kwargs)

    def _free_memory(self, ep_rew, ep_len, ep_info):
        """ Deallocate memory. """
        self.episode_rewards = [ep_rew]
        self.episode_lengths = [ep_len]
        self.episode_times = [time.time() - self.t_start]
        self.file_handler.seek(0)
        self.logger = csv.DictWriter(self.file_handler,
                                     fieldnames=("r", "l", "t") + self.reset_keywords + self.info_keywords)
        self.logger.writeheader()
        self.logger.writerow(ep_info)
        self.file_handler.flush()
        self.file_handler.truncate()
        self.file_size = 1

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        """
        Step the environment with the given action. Reset file, when number of lines is bigger than max size.

        :param action: the action
        :return: observation, reward, done, information
        """

        # ToDo: Czy moje środowiska nie wymagają resetu.
        if self.needs_reset:
            self.reset()
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
                self._free_memory(ep_rew, ep_len, ep_info)

            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, done, info

    def close(self) -> None:
        """
        Closes the environment
        """
        super(CyclicMonitor, self).close()
        if self.file_handler is not None:
            self.file_handler.close()

    def get_total_steps(self) -> int:
        """
        Returns the total number of timesteps

        :return:
        """
        return self.total_steps

    def get_episode_rewards(self) -> List[float]:
        """
        Returns the rewards of all the episodes

        :return:
        """
        return self.episode_rewards

    def get_episode_lengths(self) -> List[int]:
        """
        Returns the number of timesteps of all the episodes

        :return:
        """
        return self.episode_lengths

    def get_episode_times(self) -> List[float]:
        """
        Returns the runtime in seconds of all the episodes

        :return:
        """
        return self.episode_times