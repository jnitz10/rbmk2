from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Protocol,
    SupportsFloat,
    Tuple,
    Union
)

import gymnasium as gym
import numpy as np
import torch


GymObs = Union[Tuple, Dict[str, Any], np.ndarray, int]
GymResetReturn = Tuple[GymObs, Dict]
GymStepReturn = Tuple[GymObs, float, bool, bool, Dict]
AtariStepReturn = Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]
AtariFrames = np.ndarray
AtariActions = np.ndarray
AtariRewards = np.ndarray
AtariDones = np.ndarray
TensorDict = Dict[str, torch.Tensor]
PyTorchObs = Union[torch.Tensor, TensorDict]
NumpyObs = np.ndarray
Action = int
Transition = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
BufferSample = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]

