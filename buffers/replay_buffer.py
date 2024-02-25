import numpy as np
import random
from gymnasium.spaces import Box
from collections import deque
from utils.type_aliases import Transition, BufferSample
from typing import List, Deque, Tuple
import sys

from buffers.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer:

    def __init__(self,
                 obs_space: Box,
                 size: int,
                 batch_size: int = 32,
                 n_envs: int = 1,
                 n_step: int = 3,
                 gamma: float = 0.99,
                 alpha: float = 0.5,
                 prior_eps: float = 1e-6
                 ):
        self.max_size = max(size // n_envs, 1)
        self.obs_dims = obs_space.shape[1:]
        self.obs_buf = np.zeros((self.max_size, *self.obs_dims), dtype=np.uint8)
        self.actions_buf = np.zeros([self.max_size], dtype=np.int16)
        self.rewards_buf = np.zeros([self.max_size], dtype=np.float32)
        self.next_obs_buf = np.zeros((self.max_size, *self.obs_dims), dtype=np.uint8)
        self.dones_buf = np.zeros([self.max_size], dtype=np.bool_)
        self.mem_ctr = 0
        self.batch_size = batch_size
        self.n_envs = n_envs

        # N-Step
        self.n_step_buffer: Deque[Transition] = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

        # PER
        assert alpha >= 0
        self.alpha = alpha
        self.max_priority = 1
        self.tree_ptr = 0
        self.prior_eps = prior_eps

        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(self, transition: Transition):
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.n_step:
            return

        rewards, dones, next_obses = self._get_n_step_info()

        obses, actions = self.n_step_buffer[0][:2]

        for i in range(self.n_envs):
            index = self.mem_ctr % self.max_size
            self.obs_buf[index] = obses[i]
            self.actions_buf[index] = actions[i]
            self.rewards_buf[index] = rewards[i]
            self.dones_buf[index] = dones[i]
            self.next_obs_buf[index] = next_obses[i]
            self.mem_ctr += 1

        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample_batch(self, beta: float = 0.4) -> BufferSample:
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional()

        obses = self.obs_buf[indices]
        actions = self.actions_buf[indices]
        rewards = self.rewards_buf[indices]
        dones = self.dones_buf[indices]
        next_obses = self.next_obs_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return (
            obses,
            actions,
            rewards,
            dones,
            next_obses,
            weights,
            indices
        )

    def update_priorities(self, indices: List[int], loss: np.ndarray):
        assert len(indices) == len(loss)

        priorities = loss + self.prior_eps

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self) -> List[int]:
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float):
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight

    def _get_n_step_info(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        rewards, dones, next_obses = self.n_step_buffer[-1][-3:]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            t_rew, t_done, t_next_obs = transition[-3:]

            rewards = t_rew + self.gamma * rewards * (1 - t_done)
            dones = np.where(t_done, t_done, dones)
            for i in range(len(next_obses)):
                next_obses[i] = t_next_obs[i] if dones[i] else next_obses[i]

        return rewards, dones, next_obses

    def __len__(self) -> int:
        return min(self.mem_ctr, self.max_size)

    def __sizeof__(self):
        # get memory size of ReplayBuffer object
        return sum([sys.getsizeof(v) for v in self.__dict__.values()]) + sys.getsizeof(object.__sizeof__(self))
