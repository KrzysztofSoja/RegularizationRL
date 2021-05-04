import os
import csv
import gym
import numpy as np
import neptune

from typing import Tuple, Union, NoReturn, Dict, Optional, Any, List
from neptunecontrib.api.video import log_video
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common import logger


def save_in_csv(path_to_file: str, logs: Dict[Any, Any]):
    if os.path.isfile(path_to_file):
        with open(path_to_file, 'a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=logs.keys())
            writer.writerow(logs)
    else:
        with open(path_to_file, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=logs.keys())
            writer.writeheader()
            writer.writerow(logs)


class NeptuneCallback(BaseCallback):
    def __init__(self,
                 model,
                 environment_name,
                 neptune_account_name,
                 project_name,
                 experiment_name,
                 log_dir,
                 random_seed: Optional[int] = None,
                 model_parameter: Optional[Dict[str, Any]] = None,
                 comment: Optional[str] = None,
                 evaluate_freq: int = 10_000,
                 evaluate_length: int = 5,
                 verbose: int = 0,
                 video_length: int = 1000,
                 make_video_freq: int = 0,
                 tags: Optional[List[str]] = None):
        super(NeptuneCallback, self).__init__(verbose)
        self.model = model
        self.evaluate_freq = evaluate_freq
        self.log_dir = log_dir
        self.best_mean_reward = -np.inf
        self.video_length = video_length
        self.make_video_freq = make_video_freq

        self.neptune_account_name = neptune_account_name
        self.project_name = project_name
        self.experiment_name = experiment_name

        self.tags = [] if tags is None else tags
        self.evaluate_length = evaluate_length
        self.model_parameter = dict() if model_parameter is None else model_parameter
        self.comment = "" if comment is None else comment
        self.random_seed = random_seed

        self.environment_name = environment_name

    def _init_callback(self) -> None:
        neptune.init(self.neptune_account_name + '/' + self.project_name)
        neptune.create_experiment(self.experiment_name,
                                  params=self.model_parameter,
                                  description=self.comment)
        neptune.append_tags(self.model.__class__.__name__, self.environment_name, *self.tags)

        neptune.log_text('Path to local files', self.log_dir)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

    def _load_logs(self) -> Union[Tuple[float, float, float, float], NoReturn]:
        last_rewards = []
        last_ep_lengths = []

        for file_or_dir in os.listdir(self.log_dir):
            file_or_dir = os.path.join(self.log_dir, file_or_dir)
            if os.path.isfile(file_or_dir) and file_or_dir[-len('.monitor.csv'):] == '.monitor.csv':
                with open(file_or_dir, 'r') as csv_file:
                    reader = csv.DictReader(csv_file)

                    rows = list(reader)[1:]
                    if len(rows) == 0:
                        continue
                    lasts = min(4, len(rows))
                    rows = rows[-lasts:]

                    for row in rows:
                        last_rewards.append(float(row['r']))
                        last_ep_lengths.append(float(row['l']))
        if len(last_rewards) == 0:
            return

        last_rewards = np.array(last_rewards)
        last_ep_lengths = np.array(last_ep_lengths)
        mean_reward, std_reward = np.mean(last_rewards), np.std(last_rewards)
        mean_ep_lengths, std_ep_lengths = np.mean(last_ep_lengths), np.std(last_ep_lengths)
        return mean_reward, std_reward, mean_ep_lengths, std_ep_lengths

    def _make_video(self):
        video_env = DummyVecEnv([lambda: gym.make(self.environment_name)])

        video_name = self.experiment_name + '-step-' + str(self.n_calls)
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
        validate_environment = gym.make(self.environment_name)
        episode_rewards, episode_lengths = evaluate_policy(
            self.model,
            validate_environment,
            n_eval_episodes=self.evaluate_length,
            render=False,
            deterministic=True,
            return_episode_rewards=True,
        )

        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
        return mean_reward, std_reward, mean_ep_length, std_ep_length

    def _on_step(self) -> bool:

        basic_logs = self._load_logs()
        if basic_logs is None:
            return True
        else:
            basic_logs = {'train/mean_reward': basic_logs[0], 'train/std_reward': basic_logs[1],
                          'train/mean_episode_length': basic_logs[2], 'train/std_episode_length': basic_logs[3]}

        for metric_name, metric_value in basic_logs.items():
            neptune.log_metric(metric_name, metric_value)

        for metric_name, metric_value in logger.get_log_dict().items():
            neptune.log_metric(metric_name, metric_value)
        save_in_csv(os.path.join(self.log_dir, 'training.csv'), {**logger.get_log_dict(), **basic_logs})

        if self.n_calls % self.evaluate_freq == 0:
            mean_reward, std_reward, mean_ep_length, std_ep_length = self._evaluate()
            evaluate_logs = {'evaluate/mean_reward': mean_reward,
                             'evaluate/std_reward': std_reward,
                             'evaluate/mean_episode_length': mean_ep_length,
                             'evaluate/std_episode_length': std_ep_length}

            for metric_name, metric_value in evaluate_logs.items():
                neptune.log_metric(metric_name, metric_value)
            save_in_csv(os.path.join(self.log_dir, 'evaluating.csv'), evaluate_logs)

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(os.path.join(self.log_dir, "best_model.plk"))

        if self.n_calls % self.make_video_freq == 0:
            self._make_video()

        return True

    def __del__(self):
        try:
            neptune.stop()
        except neptune.exceptions.NeptuneNoExperimentContextException:
            pass

        for file_or_dir in os.listdir(self.log_dir):
            file_or_dir = os.path.join(self.log_dir, file_or_dir)
            if os.path.isfile(file_or_dir) and file_or_dir[-len('.monitor.csv'):] == '.monitor.csv':
                os.remove(file_or_dir)
