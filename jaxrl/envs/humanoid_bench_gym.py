import gym
import numpy as np
from gymnasium import spaces
import gymnasium as gymn
import jax.numpy as jnp

# Random scores for locomotion tasks (from SimbaV2)
HB_RANDOM_SCORE = {
    "h1-walk-v0": 2.377,
    "h1-stand-v0": 10.545,
    "h1-run-v0": 2.02,
    "h1-reach-v0": 260.302,
    "h1-hurdle-v0": 2.214,
    "h1-crawl-v0": 272.658,
    "h1-maze-v0": 106.441,
    "h1-sit_simple-v0": 9.393,
    "h1-sit_hard-v0": 2.448,
    "h1-balance_simple-v0": 9.391,
    "h1-balance_hard-v0": 9.044,
    "h1-stair-v0": 3.112,
    "h1-slide-v0": 3.191,
    "h1-pole-v0": 20.09,
}

# Target success scores for locomotion tasks (from SimbaV2)
# Note: SimbaV2 only defines success scores for h1-* (no hands) variants
# We map h1hand-* tasks to corresponding h1-* success scores
HB_SUCCESS_SCORE = {
    # Direct h1-* scores from SimbaV2
    "h1-walk-v0": 700.0,
    "h1-stand-v0": 800.0,
    "h1-run-v0": 700.0,
    "h1-reach-v0": 12000.0,
    "h1-hurdle-v0": 700.0,
    "h1-crawl-v0": 700.0,
    "h1-maze-v0": 1200.0,
    "h1-sit_simple-v0": 750.0,
    "h1-sit_hard-v0": 750.0,
    "h1-balance_simple-v0": 800.0,
    "h1-balance_hard-v0": 800.0,
    "h1-stair-v0": 700.0,
    "h1-slide-v0": 700.0,
    "h1-pole-v0": 700.0,

}

class HBGymnasiumVersionWrapper(gymn.Wrapper):
    """
    Simple wrapper for HumanoidBench compatibility
    """

    def __init__(self, env: gymn.Env):
        super().__init__(env)
        self.task = env.unwrapped.task

    def render(self):
        return self.task._env.mujoco_renderer.render(self.task._env.render_mode)

def make_humanoid_env(env_name: str, seed: int) -> gymn.Env:
    import humanoid_bench
    
    # Simple patch: HumanoidBench has 'rgbd_tuple' but gymnasium expects exactly 3 modes
    import gymnasium.envs.mujoco.mujoco_env
    original_init = gymnasium.envs.mujoco.mujoco_env.MujocoEnv.__init__
    
    def patched_init(self, *args, **kwargs):
        # Fix the assertion by temporarily removing 'rgbd_tuple' from metadata
        if hasattr(self, 'metadata') and 'render_modes' in self.metadata:
            if 'rgbd_tuple' in self.metadata['render_modes']:
                self.metadata['render_modes'] = ['human', 'rgb_array', 'depth_array']
        return original_init(self, *args, **kwargs)
    
    gymnasium.envs.mujoco.mujoco_env.MujocoEnv.__init__ = patched_init
    
    try:
        additional_kwargs = {"render_mode": "rgb_array"}
        if env_name == "h1hand-package-v0":
            additional_kwargs["policy_path"] = None
        env = gymn.make(env_name, **additional_kwargs)
        env = HBGymnasiumVersionWrapper(env)
        return env
    finally:
        gymnasium.envs.mujoco.mujoco_env.MujocoEnv.__init__ = original_init

class make_env_hb(gym.Env):
    def __init__(self, env_name='hb-walk', num_envs=2, seed=0, max_t=1000):
        np.random.seed(seed)
        seeds = np.random.randint(0, 1e6, (num_envs))
        self.max_t = max_t
        self.env_name = env_name
        self.hb_env_id = env_name
        self.envs = [make_humanoid_env(self.hb_env_id, seed) for seed in seeds]
        self.num_envs = len(self.envs)
        self.num_seeds = num_envs
        self.timesteps = np.zeros(self.num_envs)
        
        # Set up action and observation spaces
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
        obs, rews, terms, truns, successes = [], [], [], [], []
        self.timesteps += 1
        for timestep, env, action in zip(self.timesteps, self.envs, actions):
            ob, reward, terminated, truncated, info = env.step(action)
            term = terminated
            obs.append(ob)
            
            # Extract success information if available
            success = info.get('success', 0.0)
            
            if timestep > self.max_t:
                trun = True
            else:
                trun = truncated
                
            rews.append(reward)
            terms.append(term)
            truns.append(trun)
            successes.append(success)
            
        return np.stack(obs), np.stack(rews), np.stack(terms), np.stack(truns), np.stack(successes)

    # Suite adapters: For Humanoid-Bench, do not change composition; model predicts full next obs.
    def observation_to_model_prediction(self, observations):
        return observations

    def model_prediction_to_observation(self, cur_observations, pred_current_block):
        return pred_current_block

    def _video_fps(self):
        env = self.envs[0]
        for _ in range(12):
            try:
                if hasattr(env, "dt"):
                    dt = float(env.dt)
                    if dt > 0:
                        return max(1, int(round(1.0 / dt)))
            except Exception:
                pass
            try:
                unwrapped = getattr(env, "unwrapped", None)
                if unwrapped is not None and hasattr(unwrapped, "model") and hasattr(unwrapped.model, "opt"):
                    model_dt = float(unwrapped.model.opt.timestep)
                    frame_skip = float(getattr(unwrapped, "frame_skip", 1))
                    if model_dt > 0 and frame_skip > 0:
                        return max(1, int(round(1.0 / (model_dt * frame_skip))))
            except Exception:
                pass
            next_env = getattr(env, "env", None)
            if next_env is None:
                break
            env = next_env
        return 30

    def normalize_score(self, raw_return):
        """
        Normalize return using SimbaV2's success normalization formula:
        Success-Normalized(x) = (x - random_score) / (target_success_score - random_score)
        """
        if self.hb_env_id not in HB_RANDOM_SCORE or self.hb_env_id not in HB_SUCCESS_SCORE:
            # If normalization scores not available, return raw score
            return raw_return
        
        random_score = HB_RANDOM_SCORE[self.hb_env_id]
        success_score = HB_SUCCESS_SCORE[self.hb_env_id]
        
        # Avoid division by zero
        if success_score == random_score:
            return 0.0
            
        normalized = (raw_return - random_score) / (success_score - random_score)
        return normalized

    def random_step(self):
        actions = self.action_space.sample()
        obs, rews, terms, truns, successes = self.step(actions)
        return obs, rews, terms, truns, successes, actions
    
    def evaluate(self, agent, num_episodes=5, save_video=False, step=None):
        n_seeds = self.num_seeds
        successes = []
        returns_eval = []
        normalized_returns_eval = []
        
        for _episode in range(num_episodes):
            observations = self.reset()
            returns = np.zeros(n_seeds)
            success = np.zeros(n_seeds)
            has_terminated = np.zeros(n_seeds)
            frames = []
            
            for i in range(self.max_t):
                if save_video and _episode == 0:
                    try:
                        frames.append(self.envs[0].render())
                    except Exception:
                        pass
                
                actions = agent.sample_actions(observations, temperature=0.0)
                next_observations, rewards, terms, truns, successes_ = self.step(actions)
                success = np.maximum(success, successes_)  # Track if success was ever achieved
                returns[has_terminated == 0] += rewards[has_terminated == 0]
                has_terminated = np.maximum(has_terminated, terms)
                observations = next_observations            
                if has_terminated.all():
                    break
                    
            if save_video and _episode == 0 and len(frames) > 0:
                import wandb
                try:
                    video_array = np.array(frames)
                    # W&B expects shape (time, channels, height, width) for RGB arrays
                    if len(video_array.shape) == 4 and video_array.shape[-1] == 3:
                        video_array = np.transpose(video_array, (0, 3, 1, 2))
                    wandb.log({f"eval_video/{self.env_name}": wandb.Video(video_array, fps=self._video_fps(), format="mp4")}, step=step)
                except Exception as e:
                    print(f"Failed to log video to wandb: {e}")
                    
            successes.append(success)
            returns_eval.append(returns)
            
            # Compute normalized returns for each seed
            normalized_returns = np.array([self.normalize_score(ret) for ret in returns])
            normalized_returns_eval.append(normalized_returns)
            
        return {
            'success': np.array(successes).mean(axis=0), 
            'return': np.array(returns_eval).mean(axis=0),
            'goal': np.array(normalized_returns_eval).mean(axis=0)  # SimbaV2-style normalized score
        }
