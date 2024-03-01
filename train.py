import gymnasium as gym
from agent import Agent

num_envs = 4

agent = Agent(
    n_envs=num_envs, learn_start=80000, env_id="ALE/SpaceInvaders-v5"
)

agent.train(1000000)
