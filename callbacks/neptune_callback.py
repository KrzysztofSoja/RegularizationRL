import os
import csv
import gym
import json
import numpy as np
import neptune

from typing import Tuple, Union, NoReturn
from neptunecontrib.api.video import log_video
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv, VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.base_class import BaseAlgorithm


class NeptuneCallback(BaseCallback):
    def __init__(self,
                 model: BaseAlgorithm,
                 logs_freq: int,
                 evaluate_freq: int,
                 neptune_account_name: str,
                 project_name: str,
                 experiment_name: str,
                 eval_env: Union[gym.Env, VecEnv],
                 log_dir: str,
                 verbose: int = 0,
                 video_length: int = 1000):
        super(NeptuneCallback, self).__init__(verbose)
        self.model = model
        self.logs_freq = logs_freq
        self.evaluate_freq = evaluate_freq
        self.log_dir = log_dir
        self.best_mean_reward = -np.inf
        self.video_length = video_length

        self.neptune_account_name = neptune_account_name
        self.project_name = project_name
        self.experiment_name = experiment_name

        self.eval_env = eval_env

    def _init_callback(self) -> None:
        neptune.init(self.neptune_account_name + '/' + self.project_name)
        neptune.create_experiment(self.experiment_name)

        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

    def _load_logs(self) -> Union[Tuple[float, float, float, float], NoReturn]:
        last_rewards = []
        last_ep_lengths = []

        for file_or_dir in os.listdir(self.log_dir):
            file_or_dir = os.path.join(self.log_dir, file_or_dir)
            if os.path.isfile(file_or_dir) and file_or_dir[-len('.csv'):] == '.csv':
                with open(file_or_dir, 'r') as csv_file:
                    reader = csv.DictReader(csv_file)

                    rows = list(reader)[2:]
                    if len(rows) == 0:
                        continue
                    lasts = min(4, len(rows))
                    rows = rows[-lasts:]

                    for row in rows:
                        keys = list(row.keys())
                        last_rewards.append(float(row[keys[0]]))
                        last_ep_lengths.append(float(row[keys[1]]))
        if len(last_rewards) == 0:
            return

        last_rewards = np.array(last_rewards)
        last_ep_lengths = np.array(last_ep_lengths)
        mean_reward, std_reward = np.mean(last_rewards), np.std(last_rewards)
        mean_ep_lengths, std_ep_lengths = np.mean(last_ep_lengths), np.std(last_ep_lengths)
        return mean_reward, std_reward, mean_ep_lengths, std_ep_lengths

    def _make_video(self):
        video_name = self.experiment_name + '-step-' + str(self.n_calls)

        video_env = DummyVecEnv([lambda: self.eval_env])
        video_env = VecVideoRecorder(video_env, self.log_dir,
                                     record_video_trigger=lambda x: x == 0,
                                     video_length=self.video_length,
                                     name_prefix=video_name)

        observation = video_env.reset()
        for _ in range(self.video_length + 1):
            action, _ = self.model.predict(observation)
            observation, _, done, _ = video_env.step(action)

        video_env.close()

        path_to_video = os.path.join(self.log_dir, video_name + '-step-0-to-step-{}.mp4'.format(self.video_length))
        path_to_json = os.path.join(self.log_dir, video_name + '-step-0-to-step-{}.meta.json'.format(self.video_length))
        log_video(path_to_video)

        os.remove(path_to_video)
        os.remove(path_to_json)

    def _evaluate(self):
        episode_rewards, episode_lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=5,
            render=False,
            deterministic=True,
            return_episode_rewards=True,
        )

        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
        return mean_reward, std_reward, mean_ep_length, std_ep_length

    def _on_step(self) -> bool:
        if self.n_calls % self.logs_freq == 0:
            logs = self._load_logs()
            if logs is None:
                return True
            else:
                mean_reward, std_reward, mean_length, std_length = logs

            neptune.log_metric('mean reward from training', mean_reward)
            neptune.log_metric('std reward from training', std_reward)
            neptune.log_metric('mean length from training', mean_length)
            neptune.log_metric('std length from training', std_length)

        if self.n_calls % self.evaluate_freq == 0:
            mean_reward, std_reward, mean_ep_length, std_ep_length = self._evaluate()
            self._make_video()

            neptune.log_metric('mean reward from evaluate', mean_reward)
            neptune.log_metric('std_reward from evaluate', std_reward)
            neptune.log_metric('mean episode length from evaluate', mean_ep_length)
            neptune.log_metric('std episode length from evaluate', std_ep_length)

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(os.path.join(self.log_dir, "best_model"))

        return True
