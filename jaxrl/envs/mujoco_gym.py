import gym
import numpy as np
from gymnasium import spaces
import gymnasium as gymn

from jaxrl.envs.ant_truncated_obs import AntTruncatedObsEnv
from jaxrl.envs.humanoid_truncated_obs import HumanoidTruncatedObsEnv

# Standard Gymnasium v4 MuJoCo tasks
# These environments have built-in termination conditions:
# - HalfCheetah-v4: No early termination (only truncation at max_steps)
# - Hopper-v4: Terminates if height < 0.7 or |angle| > 0.2
# - Walker2d-v4: Terminates if height not in [0.8, 2.0] or |angle| > 1.0
# - Ant-v4: Terminates if height not in [0.2, 1.0]
# - Humanoid-v4: Terminates if z not in [1.0, 2.0]
MUJOCO_TASKS = [
    "HalfCheetah-v4",
    "Hopper-v4",
    "Walker2d-v4",
    "Ant-v4",
    "Humanoid-v4",
]

class make_env_mujoco(gym.Env):
    def __init__(self, env_name='HalfCheetah-v4', num_envs=2, seed=0, max_t=1000):
        np.random.seed(seed)
        seeds = np.random.randint(0, 1e6, (num_envs))
        self.max_t = max_t
        # For Ant and Humanoid, match the truncated-observation implementations used in InfoProp.
        if env_name == "Humanoid-v4":
            self.envs = [HumanoidTruncatedObsEnv(render_mode="rgb_array") for _ in seeds]
        elif env_name == "Ant-v4":
            self.envs = [AntTruncatedObsEnv(render_mode="rgb_array") for _ in seeds]
        else:
            self.envs = [gymn.make(env_name, render_mode="rgb_array") for _ in seeds]
        self.num_envs = len(self.envs)
        self.num_seeds = num_envs
        self.timesteps = np.zeros(self.num_envs)
        
        self.action_space = spaces.Box(low=self.envs[0].action_space.low[None].repeat(len(self.envs), axis=0),
                                       high=self.envs[0].action_space.high[None].repeat(len(self.envs), axis=0),
                                       shape=(len(self.envs), self.envs[0].action_space.shape[0]),
                                       dtype=self.envs[0].action_space.dtype)
        self.observation_space = spaces.Box(low=self.envs[0].observation_space.low[None].repeat(len(self.envs), axis=0),
                                            high=self.envs[0].observation_space.high[None].repeat(len(self.envs), axis=0),
                                            shape=(len(self.envs), self.envs[0].observation_space.shape[0]),
                                            dtype=self.envs[0].observation_space.dtype)
        self.action_dim = self.envs[0].action_space.shape[0]
        
    def _reset_idx(self, idx):
        obs, _ = self.envs[idx].reset()
        return obs
    
    def reset_where_done(self, observations, terms, truns):
        resets = np.zeros((terms.shape))
        for j, (term, trun) in enumerate(zip(terms, truns)):
            if (term == True) or (trun == True):
                observations[j], terms[j], truns[j] = self._reset_idx(j), False, False
                resets[j] = 1
                self.timesteps[j] = 0
        return observations, terms, truns, resets
    
    def reset(self):
        obs = []
        self.timesteps = np.zeros(self.num_envs)
        for env in self.envs:
            ob, _ = env.reset()
            obs.append(ob)
        return np.stack(obs)
    
    def generate_masks(self, terms, truns):
        masks = []
        for term, trun in zip(terms, truns):
            if not term or trun:
                mask = 1.0
            else:
                mask = 0.0
            masks.append(mask)
        masks = np.array(masks)
        return masks
    
    def step(self, actions):
        obs, rews, terms, truns = [], [], [], []
        self.timesteps += 1
        for timestep, env, action in zip(self.timesteps, self.envs, actions):
            ob, reward, terminated, truncated, info = env.step(action)
            obs.append(ob)
            rews.append(reward)
            # terminated: environment-specific early termination (e.g., Hopper falls)
            # truncated: time limit reached (max_episode_steps)
            terms.append(terminated)
            if timestep >= self.max_t:
                trun = True
            else:
                trun = truncated
            truns.append(trun)
        return np.stack(obs), np.stack(rews), np.stack(terms), np.stack(truns), None

    # Suite adapters: For MuJoCo, do not change composition; model predicts full next obs.
    def observation_to_model_prediction(self, observations):
        return observations

    def model_prediction_to_observation(self, cur_observations, pred_current_block):
        return pred_current_block

    def random_step(self):
        actions = self.action_space.sample()
        obs, rews, terms, truns, _ = self.step(actions)
        return obs, rews, terms, truns, None, actions
    
    def evaluate(self, agent, num_episodes=5, save_video=False, step=None):
        n_seeds = self.num_seeds
        returns_eval = []
        for _episode in range(num_episodes):
            observations = self.reset()
            returns = np.zeros(n_seeds)
            # Track which parallel environments have finished their episode.
            has_terminated = np.zeros(n_seeds, dtype=bool)
            frames = []
            for i in range(self.max_t):
                if save_video and _episode == 0:
                    try:
                        frames.append(self.envs[0].render())
                    except Exception:
                        pass
                
                actions = agent.sample_actions(observations, temperature=0.0)
                next_observations, rewards, terms, truns, _ = self.step(actions)
                # Only accumulate rewards for environments that are still running.
                active = ~has_terminated
                returns[active] += rewards[active]
                # Mark episodes as finished when either terminated or truncated.
                has_terminated = np.logical_or(has_terminated, np.logical_or(terms, truns))
                observations = next_observations
                # If all parallel envs have finished, stop this evaluation episode early.
                if has_terminated.all():
                    break
                    
            if save_video and _episode == 0 and len(frames) > 0:
                import wandb
                try:
                    video_array = np.array(frames)
                    if len(video_array.shape) == 4 and video_array.shape[-1] == 3:
                        video_array = np.transpose(video_array, (0, 3, 1, 2))
                    env_name = getattr(self, 'env_name', 'mujoco_env')
                    wandb.log({f"eval_video/{env_name}": wandb.Video(video_array, fps=30, format="mp4")}, step=step)
                except Exception as e:
                    print(f"Failed to log video to wandb: {e}")
                    
            returns_eval.append(returns)
        return {'return': np.array(returns_eval).mean(axis=0)}

