import gym
import numpy as np
from gymnasium import spaces
import gymnasium as gymn
import myosuite
import jax.numpy as jnp

MYOSUITE_TASKS = {
    'myo-test' : 'myoElbowPose1D6MRandom-v0',
	'myo-reach': 'myoHandReachFixed-v0',
	'myo-reach-hard': 'myoHandReachRandom-v0',
	'myo-pose': 'myoHandPoseFixed-v0',
	'myo-pose-hard': 'myoHandPoseRandom-v0',
	'myo-obj-hold': 'myoHandObjHoldFixed-v0',
	'myo-obj-hold-hard': 'myoHandObjHoldRandom-v0',
	'myo-key-turn': 'myoHandKeyTurnFixed-v0',
	'myo-key-turn-hard': 'myoHandKeyTurnRandom-v0',
	'myo-pen-twirl': 'myoHandPenTwirlFixed-v0',
	'myo-pen-twirl-hard': 'myoHandPenTwirlRandom-v0',
}

class make_env_myo(gym.Env):
    def __init__(self, env_name='myo-test', num_envs=2, seed=0, max_t=100):
        np.random.seed(seed)
        seeds = np.random.randint(0,1e6,(num_envs))
        self.max_t = max_t
        self.envs = [gymn.make(MYOSUITE_TASKS[env_name]) for seed in seeds]
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
        obs, rews, terms, truns, goals = [], [], [], [], []
        self.timesteps += 1
        for timestep, env, action in zip(self.timesteps, self.envs, actions):
            ob, reward, _, _, info = env.step(action)
            term = False
            obs.append(ob)
            goal = info['solved']
            #if timestep == self.max_t or info['solved'] == 1:
            if timestep == self.max_t:
                trun = True
            else:
                trun = False
            rews.append(reward)
            terms.append(term)
            truns.append(trun)
            goals.append(goal)
        return np.stack(obs), np.stack(rews), np.stack(terms), np.stack(truns), np.stack(goals)

    # Suite adapters: For MyoSuite, do not change composition; model predicts full next obs.
    def observation_to_model_prediction(self, observations):
        return observations

    def model_prediction_to_observation(self, cur_observations, pred_current_block):
        return pred_current_block

    def random_step(self):
        actions = self.action_space.sample()
        obs, rews, terms, truns, goals = self.step(actions)
        return obs, rews, terms, truns, goals, actions
    
    def evaluate(self, agent, num_episodes=5, save_video=False, step=None):
        n_seeds = self.num_seeds
        goals = []
        returns_eval = []
        for _episode in range(num_episodes):
            observations = self.reset()
            returns = np.zeros(n_seeds)
            goal = np.zeros(n_seeds)
            frames = []
            for i in range(self.max_t): # CHANGE?
                if save_video and _episode == 0:
                    try:
                        if hasattr(self.envs[0], 'render_mode'):
                            frames.append(self.envs[0].render())
                        else:
                            frames.append(self.envs[0].render(mode='rgb_array'))
                    except Exception:
                        pass
                
                actions = agent.sample_actions(observations, temperature=0.0)
                next_observations, rewards, terms, truns, goals_ = self.step(actions)
                goal += goals_  # Track if solved at any timestep
                returns += rewards
                observations = next_observations            
                
            if save_video and _episode == 0 and len(frames) > 0:
                import wandb
                try:
                    video_array = np.array(frames)
                    if len(video_array.shape) == 4 and video_array.shape[-1] == 3:
                        video_array = np.transpose(video_array, (0, 3, 1, 2))
                    env_name = getattr(self, 'env_name', 'myo_env')
                    wandb.log({f"eval_video/{env_name}": wandb.Video(video_array, fps=30, format="mp4")}, step=step)
                except Exception as e:
                    print(f"Failed to log video to wandb: {e}")
                    
            goal = (goal > 0).astype(float)  # Binary: 1.0 if solved at any point, 0.0 otherwise
            goals.append(goal)
            returns_eval.append(returns)
        return {'goal': np.array(goals).mean(axis=0), 'return': np.array(returns_eval).mean(axis=0)}
    
