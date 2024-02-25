import gymnasium as gym
from agent import Agent

num_envs = 4

agent = Agent(n_envs=num_envs, learn_start=20000)

agent.train(100000)
