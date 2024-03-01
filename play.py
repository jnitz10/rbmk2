import gymnasium as gym
from agent import Agent

num_envs = 4

agent = Agent(env_id='ALE/SpaceInvaders-v5', n_envs=num_envs, learn_start=20000)

agent.play(20)
