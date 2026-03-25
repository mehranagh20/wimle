import copy

import gym
import numpy as np
from gym.spaces import Box, Dict

class SinglePrecision(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        source_space = self.env.observation_space

        if isinstance(source_space, Box):
            obs_space = source_space
            self.observation_space = Box(obs_space.low, obs_space.high,
                                         obs_space.shape)
        elif isinstance(source_space, Dict):
            obs_spaces = copy.copy(source_space.spaces)
            for k, v in obs_spaces.items():
                obs_spaces[k] = Box(v.low, v.high, v.shape)
            self.observation_space = Dict(obs_spaces)
        else:
            raise NotImplementedError

    def observation(self, observation: np.ndarray) -> np.ndarray:
        if isinstance(observation, np.ndarray):
            return observation.astype(np.float32)
        elif isinstance(observation, dict):
            observation = copy.copy(observation)
            for k, v in observation.items():
                observation[k] = v.astype(np.float32)
            return observation
