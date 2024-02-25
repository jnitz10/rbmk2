import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack, TransformReward, RecordEpisodeStatistics
from stable_baselines3.common.atari_wrappers import FireResetEnv
import numpy as np


def make_atari_env(env_id='ALE/Breakout-v5', is_play=False, is_record=False):
    if is_play:
        env = gym.make(env_id, frameskip=1, render_mode='human')
    elif is_record:
        env = gym.make(env_id, frameskip=1, render_mode='rgb_array')
    else:
        env = gym.make(env_id, frameskip=1)
    processed_env = AtariPreprocessing(env, terminal_on_life_loss=True)
    transformed_rewards = TransformReward(processed_env, np.sign)
    stacked_env = FrameStack(transformed_rewards, 4)
    fire_env = FireResetEnv(stacked_env)

    return fire_env


def make_venv(env_id='ALE/Breakout-v5', n_envs: int = 1):
    venv = gym.vector.SyncVectorEnv([lambda: make_atari_env(env_id=env_id)] * n_envs)
    venv = RecordEpisodeStatistics(venv)

    return venv
