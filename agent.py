from collections import deque
from datetime import datetime

from torch.nn.utils import clip_grad_norm_

from buffers import ReplayBuffer
from network import DQN
import torch
import torch.optim as optim
import torch.utils.tensorboard as tb
import numpy as np
import os
from typing import Optional, Tuple, Union
from tqdm import tqdm

from utils import make_venv, make_atari_env, make_record_env
from utils.type_aliases import Transition, NumpyObs, PyTorchObs


class Agent:
    """
    Initializes rainbow DQN agent.

    Parameters:
    - env_id (str): Identifier for the environment to use (default: 'ALE/Breakout-v5').
    - n_envs (int): Number of parallel environments to run (default: 1).
    - memory_size (int): Size of the replay buffer (default: 1,000,000).
    - batch_size (int): Number of experiences to sample from memory per update (default: 32).
    - n_step (int): Number of steps for n-step returns (default: 3).
    - target_update (int): Frequency of target network updates (default: 8,000 steps).
    - gamma (float): Discount factor for future rewards (default: 0.99).
    - chkpt_dir (str): Directory to save model checkpoints (default: 'tmp/').
    - log_dir (str): Directory for TensorBoard logs (default: 'runs/').
    - optimizer_params (Optional[dict]): Parameters for the optimizer (default: None).
    - learn_start (int): Number of steps before starting learning (default: 20,000)
    - alpha (float): PER alpha parameter, controlling sampling bias (default: 0.2).
    - beta (float): PER beta parameter, controlling importance-sampling weight (default: 0.6).
    - prior_eps (float): Small value to ensure no experience has zero priority (default: 1e-6).
    - v_min (float): Minimum value prediction in Categorical DQN (default: -10.0).
    - v_max (float): Maximum value prediction in Categorical DQN (default: 10.0).
    - n_atoms (int): Number of atoms in Categorical DQN (default: 51).

    Attributes:
    - venv: Vectorized environments based on the given `env_id` and `n_envs`.
    - obs_space: Observation space of the environment.
    - action_dim: Dimension of the action space.
    - n_envs: Number of parallel environments.
    - memory_size: Size of the replay buffer.
    - batch_size: Batch size for sampling from the replay buffer.
    - n_step: The number of steps to look ahead for n-step returns.
    - target_update: Frequency (in steps) of updating the target network.
    - gamma: Discount factor for future rewards.
    - is_test: Flag to indicate whether the agent is in training or testing mode.
    - learn_step_counter: Counter for the number of learning steps taken.
    - episodes: Counter for the number of episodes processed.
    - device: The device (CPU or CUDA) on which tensors will be allocated.
    - chkpt_dir: Directory path for saving model checkpoints.
    - log_dir: Directory path for TensorBoard logging.
    - dqn_chkpt: Path for saving the DQN model checkpoints.
    - target_chkpt: Path for saving the target network checkpoints.
    - beta: Beta parameter for PER, adjusting the importance-sampling weight.
    - prior_eps: Epsilon parameter for PER to ensure no zero-priority.
    - memory: The replay buffer with prioritized experience replay.
    - v_min: Minimum possible value for value distribution in Categorical DQN.
    - v_max: Maximum possible value for value distribution in Categorical DQN.
    - n_atoms: Number of atoms for value distribution in Categorical DQN.
    - support: The support of the value distribution in Categorical DQN.
    - delta_z: The spacing between atoms in the value distribution.
    - dqn: The current DQN model being trained.
    - target_net: The target DQN model for stable Q-value estimation.
    - optimizer: The optimizer used for training the DQN model.
    - tb_writer: TensorBoard writer for logging training metrics.
    """

    def __init__(
        self,
        env_id: str = "ALE/Breakout-v5",
        n_envs: int = 1,
        memory_size: int = 1000000,
        batch_size: int = 32,
        n_step: int = 3,
        target_update: int = 8000,
        gamma: float = 0.99,
        chkpt_dir: str = "tmp/",
        log_dir: str = "runs/",
        optimizer_params: Optional[dict] = None,
        learn_start: int = 20000,
        # PER params
        alpha: float = 0.2,
        beta: float = 0.6,
        beta_amortization: int = 3000000,
        prior_eps: float = 1e-6,
        # Categorical DQN params
        v_min: float = -10.0,
        v_max: float = 10.0,
        n_atoms: int = 51,
    ):
        self.venv = make_venv(env_id, n_envs)
        self.env_id = env_id
        self.obs_space = self.venv.observation_space
        self.action_dim = self.venv.action_space[0].n
        self.n_envs = n_envs
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.n_step = n_step
        self.target_update = target_update
        self.gamma = gamma
        self.is_test = False
        self.learn_step_counter = 0
        self.episodes = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learn_start = learn_start

        # Checkpoint and log directories
        self.chkpt_dir = os.path.join(chkpt_dir, env_id.lstrip("ALE/"))
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        self.log_dir = os.path.join(log_dir, current_time + "_" + env_id.lstrip("ALE/"))
        self.dqn_chkpt = os.path.join(self.chkpt_dir, "dqn")
        self.target_chkpt = os.path.join(self.chkpt_dir, "target")
        self.video_dir = os.path.join("videos/", env_id.lstrip("ALE/"))

        # PER
        # TODO: decide on how to implement saving/loading of buffer
        self.beta = beta
        self.beta_amortization = beta_amortization
        self.prior_eps = prior_eps
        self.memory = ReplayBuffer(
            self.obs_space,
            n_envs=n_envs,
            size=memory_size,
            batch_size=batch_size,
            alpha=alpha,
            gamma=gamma,
            n_step=n_step,
            prior_eps=prior_eps,
        )

        # Categorical
        self.v_min = v_min
        self.v_max = v_max
        self.n_atoms = n_atoms
        self.support = torch.linspace(self.v_min, self.v_max, n_atoms).to(self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)

        # Network
        self.dqn = DQN(out_dim=self.action_dim, n_atoms=n_atoms).to(self.device)
        self.target_net = DQN(out_dim=self.action_dim, n_atoms=n_atoms).to(self.device)
        self.target_net.load_state_dict(self.dqn.state_dict())
        self.target_net.eval()
        optimizer_params = (
            optimizer_params if optimizer_params else {"lr": 6.25e-5, "eps": 1.5e-4}
        )
        self.optimizer = optim.Adam(self.dqn.parameters(), **optimizer_params)
        self.tb_writer = tb.SummaryWriter(self.log_dir)

        self.losses = deque(maxlen=100)
        self.ep_rewards = deque(maxlen=100)
        self.startup()

    def store_transition(self, transition: Transition):
        self.memory.store(transition)

    def select_actions(self, observations: Union[NumpyObs, PyTorchObs]) -> np.ndarray:
        """add batch dimension if selecting for single obs"""
        with torch.no_grad():
            if isinstance(observations, torch.Tensor):
                dist = self.dqn(observations)
            else:
                dist = self.dqn(torch.from_numpy(observations).to(self.device))
            actions = (dist * self.support).sum(2).argmax(1)
            return actions.cpu().numpy()

    def step_environment(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        next_obses, rewards, terminateds, truncateds, infos = self.venv.step(actions)
        dones = terminateds | truncateds
        if "episode" in infos and len(self.memory) > self.learn_start:
            ep_rewards = infos["episode"]["r"]
            completed_eps = infos["_episode"]
            eps = ep_rewards[completed_eps]
            for ep in eps:
                self.episodes += 1
                self.ep_rewards.append(ep)
                self.tb_writer.add_scalar("Episode Reward", ep, self.episodes)

        return next_obses, rewards, dones

    def populate_buffer(self):
        with tqdm(total=self.learn_start, desc="Filling buffer") as pbar:
            obses, _ = self.venv.reset()
            while len(self.memory) < self.learn_start:
                actions = self.select_actions(obses)
                next_obses, rewards, dones = self.step_environment(actions)
                transition = (obses, actions, rewards, dones, next_obses)
                self.store_transition(transition)
                obses = next_obses
                self.dqn.reset_noise()
                if len(self.memory) % 1000 == 0:
                    pbar.update(1000)
                    pbar.set_postfix_str(f"{len(self.memory)} frames so far...")
                pbar.update(len(self.memory) - pbar.n)

    def train(self, n_learning_steps):
        obses, _ = self.venv.reset()

        if len(self.memory) < self.learn_start:
            self.populate_buffer()

        pbar = tqdm(range(n_learning_steps), desc="Training Progress")

        for _ in pbar:
            actions = self.select_actions(obses)
            next_obses, rewards, dones = self.step_environment(actions)
            transition = (obses, actions, rewards, dones, next_obses)
            self.store_transition(transition)
            obses = next_obses

            fraction = min(self.learn_step_counter / self.beta_amortization, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            loss = self.learn()
            self.losses.append(loss)

            self.learn_step_counter += 1

            if self.learn_step_counter % 500 == 0:
                average_loss = sum(self.losses) / max(len(self.losses), 1)
                average_ep_rew = sum(self.ep_rewards) / max(len(self.ep_rewards), 1)
                pbar.set_postfix(
                    avg_loss="{:.2f}".format(average_loss),
                    avg_ep_reward="{:.2f}".format(average_ep_rew),
                    refresh=True,
                )

            if self.learn_step_counter % 2000 == 0:
                self.save_checkpoint()

            if self.learn_step_counter % 50000 == 0:
                self.record(10)

            if self.learn_step_counter % self.target_update == 0:
                self.update_target_network()

    def learn(self):
        obses, actions, rewards, dones, next_obses, weights, indices = (
            self.memory.sample_batch(self.beta)
        )
        torch.autograd.set_detect_anomaly(True)

        obses = torch.tensor(obses, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device).reshape(-1, 1)
        dones = torch.tensor(dones, device=self.device).reshape(-1, 1)
        next_obses = torch.tensor(next_obses, device=self.device)
        weights = torch.tensor(weights, device=self.device)

        log_probs = self.dqn(obses, log=True)
        log_prob_actions = log_probs[range(self.batch_size), actions.long()]

        with torch.no_grad():

            # best next actions chosen by dqn
            next_obs_probs = self.dqn(next_obses)
            next_obs_dist = self.support.expand_as(next_obs_probs) * next_obs_probs
            best_indices = next_obs_dist.sum(2).argmax(1)
            # target gets value probs
            target_obs_probs = self.target_net(next_obses)
            target_actions = target_obs_probs[range(self.batch_size), best_indices]

            # target_support = Tz, apply Bellman operator to Tz
            target_support = rewards + (1 - dones.float()) * (
                self.gamma**self.n_step
            ) * self.support.unsqueeze(0)
            target_support = target_support.clamp(min=self.v_min, max=self.v_max)
            # L2 projection of target_support onto support
            normalized_support_indices = (target_support - self.v_min) / self.delta_z
            lower = normalized_support_indices.floor().to(torch.int64)
            upper = normalized_support_indices.ceil().to(torch.int64)
            # handle edge case when lower = target = upper (target is int)
            lower[(upper > 0) * (lower == upper)] -= 1
            upper[(lower < (self.n_atoms - 1)) * (lower == upper)] += 1

            offset = (
                torch.linspace(0, (self.batch_size - 1) * self.n_atoms, self.batch_size)
                .long()
                .unsqueeze(1)
                .expand(self.batch_size, self.n_atoms)
                .to(self.device)
            )

            proj_dist = torch.zeros(
                (self.batch_size, self.n_atoms), device=self.device
            ).float()

            proj_dist.view(-1).index_add_(
                0,
                (lower + offset).view(-1),
                (target_actions * (upper.float() - normalized_support_indices)).view(
                    -1
                ),
            )
            proj_dist.view(-1).index_add_(
                0,
                (upper + offset).view(-1),
                (target_actions * (normalized_support_indices - lower.float())).view(
                    -1
                ),
            )

        elementwise_loss = -torch.sum(proj_dist * log_prob_actions, 1)
        self.dqn.zero_grad()

        loss = (weights * elementwise_loss).mean()

        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        self.tb_writer.add_scalar("Loss", loss.detach(), self.learn_step_counter)
        self.memory.update_priorities(indices, elementwise_loss.detach().cpu().numpy())
        # reset noise in networks
        self.dqn.reset_noise()
        self.target_net.reset_noise()

        return loss.detach().cpu().numpy()

    def play(self, n_episodes):
        play_env = make_atari_env(env_id=self.env_id, is_play=True)

        obs, _ = play_env.reset()

        for i in range(n_episodes):
            done = False
            while not done:
                action = self.select_actions(
                    torch.from_numpy(np.array(obs)).unsqueeze(0).to(self.device)
                )[0]
                next_obs, reward, terminated, truncated, info = play_env.step(action)
                done = terminated or truncated
                obs = next_obs
            obs, _ = play_env.reset()
        play_env.close()

    def record(self, n_episodes):
        video_folder = os.path.join(
            self.video_dir, "step_" + str(self.learn_step_counter)
        )
        env = make_record_env(env_id=self.env_id, video_folder=video_folder)

        for i in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action = self.select_actions(
                    torch.from_numpy(np.array(obs)).unsqueeze(0).to(self.device)
                )[0]
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                obs = next_obs
        env.close()

    def update_target_network(self):
        self.target_net.load_state_dict(self.dqn.state_dict())

    def save_checkpoint(self):
        torch.save(
            {
                "q_eval": self.dqn.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "learn_step_counter": self.learn_step_counter,
                "beta": self.beta,
                "episodes": self.episodes,
            },
            self.dqn_chkpt,
        )
        torch.save(
            {
                "target_net": self.target_net.state_dict(),
            },
            self.target_chkpt,
        )

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.dqn.load_state_dict(torch.load(self.dqn_chkpt)["q_eval"])
        self.optimizer.load_state_dict(torch.load(self.dqn_chkpt)["optimizer"])
        self.learn_step_counter = torch.load(self.dqn_chkpt)["learn_step_counter"]
        self.beta = torch.load(self.dqn_chkpt)["beta"]
        self.target_net.load_state_dict(torch.load(self.target_chkpt)["target_net"])
        self.episodes = torch.load(self.dqn_chkpt)["episodes"]
        print(
            f"... checkpoint loaded ... learn_step_counter: {self.learn_step_counter}"
        )

    def startup(self):
        if not os.path.exists(self.chkpt_dir):
            os.mkdir(self.chkpt_dir)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        if os.path.exists(self.dqn_chkpt) and os.path.exists(self.target_chkpt):
            print("Checkpoint found...")
            self.load_checkpoint()
        else:
            print("No checkpoint found...")

    def test_directory_setup(self):
        test_chkpt_path = os.path.join(self.chkpt_dir, "test_chkpt_dir.txt")

        test_tb_path = os.path.join(self.log_dir, "test_log_dir.txt")

        with open(test_chkpt_path, "w") as f:
            f.write("Testing checkpoint directory.")

        with open(test_tb_path, "w") as f:
            f.write("Testing logs directory.")
