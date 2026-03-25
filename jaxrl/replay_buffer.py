import gym
import numpy as np

import os
import pickle
import collections
import jax.numpy as jnp

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'dones', 'next_observations', 'coefs'])


class ParallelReplayBuffer:
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int,
                 capacity: int, num_seeds: int):
        self.num_seeds = num_seeds
        self.observations = np.empty((num_seeds, capacity, observation_space.shape[-1]), dtype=observation_space.dtype)
        self.actions = np.empty((num_seeds, capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((num_seeds, capacity, ), dtype=np.float32)
        self.masks = np.empty((num_seeds, capacity, ), dtype=np.float32)
        self.dones_float = np.empty((num_seeds, capacity, ), dtype=np.float32)
        self.next_observations = np.empty((num_seeds, capacity, observation_space.shape[-1]), dtype=observation_space.dtype)
        self.coefs = np.empty((num_seeds, capacity, ), dtype=np.float32)
        self.size = 0
        self.insert_index = 0
        self.capacity = capacity
        self.n_parts = 4
        
        
    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: float, mask: float, done_float: float,
               next_observation: np.ndarray, coef: float=1.0):
        self.observations[:, self.insert_index] = observation
        self.actions[:, self.insert_index] = action
        self.rewards[:, self.insert_index] = reward
        self.masks[:, self.insert_index] = mask
        self.dones_float[:, self.insert_index] = done_float
        self.next_observations[:, self.insert_index] = next_observation
        self.coefs[:, self.insert_index] = coef
        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def _get_valid_indices(self, batch_size: int, num_batches: int = 0, size_to_consider: int = None) -> np.ndarray:
        if size_to_consider is None:
            size_to_consider = self.size

        if num_batches >= 1:
            return np.random.randint(size_to_consider, size=(num_batches, batch_size))
        else:
            return np.random.randint(size_to_consider, size=(batch_size))

    def sample_parallel(self, batch_size: int) -> Batch:
        indx = self._get_valid_indices(batch_size)
        return Batch(observations=self.observations[:, indx],
                    actions=self.actions[:, indx],
                    rewards=self.rewards[:, indx],
                    masks=self.masks[:, indx],
                    dones=self.dones_float[:, indx],
                    next_observations=self.next_observations[:, indx],
                    coefs=self.coefs[:, indx])

    def sample_parallel_multibatch(self, batch_size: int, num_batches: int) -> Batch:
        indxs = self._get_valid_indices(batch_size, num_batches)
        return Batch(observations=self.observations[:, indxs],
                    actions=self.actions[:, indxs],
                    rewards=self.rewards[:, indxs],
                    masks=self.masks[:, indxs],
                    dones=self.dones_float[:, indxs],
                    next_observations=self.next_observations[:, indxs],
                    coefs=self.coefs[:, indxs])

    def get_all_last_samples(self, num_consider: int, batch_size: int, num_batches: int) -> Batch:
        """Return the last num_samples samples."""
        if num_consider < 0:
            num_consider = self.size
        num_consider = min(num_consider, self.size)
        init_indices = np.arange(self.size - num_consider, self.size)
        indices = self._get_valid_indices(batch_size, num_batches, size_to_consider=num_consider)
        indices = init_indices[indices]
        return Batch(
            observations=self.observations[:, indices],
            actions=self.actions[:, indices],
            rewards=self.rewards[:, indices],
            masks=self.masks[:, indices],
            dones=self.dones_float[:, indices],
            next_observations=self.next_observations[:, indices],
            coefs=self.coefs[:, indices]
        )

    def get_last_window(self, num_last: int) -> Batch:
        if self.size == 0:
            raise ValueError("Replay buffer is empty.")

        window = min(num_last, self.size)
        start = self.size - window
        indices = np.arange(start, self.size)
        return Batch(
            observations=self.observations[:, indices],
            actions=self.actions[:, indices],
            rewards=self.rewards[:, indices],
            masks=self.masks[:, indices],
            dones=self.dones_float[:, indices],
            next_observations=self.next_observations[:, indices],
            coefs=self.coefs[:, indices],
        )
    
    def sample_parallel_last(self, num_consider: int, batch_size: int) -> Batch:
        """Return the last num_samples samples."""
        if num_consider < 0:
            num_consider = self.size
        num_consider = min(num_consider, self.size)
        init_indices = np.arange(self.size - num_consider, self.size)
        indices = self._get_valid_indices(batch_size, size_to_consider=num_consider)
        indices = init_indices[indices]
        return Batch(
            observations=self.observations[:, indices],
            actions=self.actions[:, indices],
            rewards=self.rewards[:, indices],
            masks=self.masks[:, indices],
            dones=self.dones_float[:, indices],
            next_observations=self.next_observations[:, indices],
            coefs=self.coefs[:, indices]
        )
    
    def concat_batches(self, batch1: Batch, batch2: Batch) -> Batch:
        # Concatenate the batches
        concatenated_obs = np.concatenate([batch1.observations, batch2.observations], axis=1)
        concatenated_actions = np.concatenate([batch1.actions, batch2.actions], axis=1)
        concatenated_rewards = np.concatenate([batch1.rewards, batch2.rewards], axis=1)
        concatenated_masks = np.concatenate([batch1.masks, batch2.masks], axis=1)
        concatenated_dones = np.concatenate([batch1.dones, batch2.dones], axis=1)
        concatenated_next_obs = np.concatenate([batch1.next_observations, batch2.next_observations], axis=1)
        concatenated_coefs = np.concatenate([batch1.coefs, batch2.coefs], axis=1)
        
        # Shuffle along axis=1
        shuffle_indices = np.random.permutation(concatenated_obs.shape[1])
        
        return Batch(
            concatenated_obs[:, shuffle_indices],
            concatenated_actions[:, shuffle_indices],
            concatenated_rewards[:, shuffle_indices],
            concatenated_masks[:, shuffle_indices],
            concatenated_dones[:, shuffle_indices],
            concatenated_next_obs[:, shuffle_indices],
            concatenated_coefs[:, shuffle_indices],
        )

    def save(self, save_dir: str):
        data_path = os.path.join(save_dir, 'buffer')
        # because of memory limits, we will dump the buffer into multiple files
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        chunk_size = self.capacity // self.n_parts

        for i in range(self.n_parts):
            data_chunk = [
                self.observations[:, i*chunk_size : (i+1)*chunk_size],
                self.actions[:, i*chunk_size : (i+1)*chunk_size],
                self.rewards[:, i*chunk_size : (i+1)*chunk_size],
                self.masks[:, i*chunk_size : (i+1)*chunk_size],
                self.dones_float[:, i*chunk_size : (i+1)*chunk_size],
                self.next_observations[:, i*chunk_size : (i+1)*chunk_size],
                self.coefs[:, i*chunk_size : (i+1)*chunk_size]
            ]

            data_path_splitted = data_path.split('buffer')
            data_path_splitted[-1] = f'_chunk_{i}{data_path_splitted[-1]}'
            data_path_chunk = 'buffer'.join(data_path_splitted)
            pickle.dump(data_chunk, open(data_path_chunk, 'wb'))
        # Save also size and insert_index
        pickle.dump((self.size, self.insert_index), open(os.path.join(save_dir, 'buffer_info'), 'wb'))

    def load(self, save_dir: str):
        data_path = os.path.join(save_dir, 'buffer')
        chunk_size = self.capacity // self.n_parts

        for i in range(self.n_parts):
            print(f"Loading chunk {i} of {self.n_parts}")
            data_path_splitted = data_path.split('buffer')
            data_path_splitted[-1] = f'_chunk_{i}{data_path_splitted[-1]}'
            data_path_chunk = 'buffer'.join(data_path_splitted)
            data_chunk = pickle.load(open(data_path_chunk, "rb"))

            self.observations[:, i*chunk_size : (i+1)*chunk_size], \
            self.actions[:, i*chunk_size : (i+1)*chunk_size], \
            self.rewards[:, i*chunk_size : (i+1)*chunk_size], \
            self.masks[:, i*chunk_size : (i+1)*chunk_size], \
            self.dones_float[:, i*chunk_size : (i+1)*chunk_size], \
            self.next_observations[:, i*chunk_size : (i+1)*chunk_size], \
            self.coefs[:, i*chunk_size : (i+1)*chunk_size] = data_chunk
        self.size, self.insert_index = pickle.load(open(os.path.join(save_dir, 'buffer_info'), 'rb'))