import gym
import numpy as np
from jaxrl.envs.single_precision import SinglePrecision
import copy
import ctypes
import sys
from typing import Dict, Optional, OrderedDict, Tuple
from dm_control import suite
from dm_env import specs
from gym import core, spaces

TimeStep = Tuple[np.ndarray, float, bool, dict]
_EGL_PATCH_APPLIED = False


def _patch_dm_control_egl_py312():
    """Patch dm_control EGL init for Python 3.12 ctypes compatibility."""
    global _EGL_PATCH_APPLIED
    if _EGL_PATCH_APPLIED or sys.version_info < (3, 12):
        return

    try:
        from dm_control._render.pyopengl import egl_renderer
        EGL = egl_renderer.EGL

        def _patched_platform_init(self, unused_max_width, unused_max_height):
            num_configs = ctypes.c_long(0)
            config_size = 1
            configs = (EGL.EGLConfig * config_size)()
            EGL.eglReleaseThread()
            EGL.eglChooseConfig(
                egl_renderer.EGL_DISPLAY,
                egl_renderer.EGL_ATTRIBUTES,
                configs,
                config_size,
                num_configs,
            )
            if num_configs.value < 1:
                raise RuntimeError(
                    'EGL failed to find a framebuffer configuration that matches the '
                    f'desired attributes: {egl_renderer.EGL_ATTRIBUTES}'
                )
            EGL.eglBindAPI(EGL.EGL_OPENGL_API)
            self._context = EGL.eglCreateContext(
                egl_renderer.EGL_DISPLAY, configs[0], EGL.EGL_NO_CONTEXT, None
            )
            if not self._context:
                raise RuntimeError('Cannot create an EGL context.')

        egl_renderer.EGLContext._platform_init = _patched_platform_init
        _EGL_PATCH_APPLIED = True
    except Exception:
        # If patching fails, dm_control uses its default path.
        pass

def dmc_spec2gym_space(spec):
    if isinstance(spec, OrderedDict):
        spec = copy.copy(spec)
        for k, v in spec.items():
            spec[k] = dmc_spec2gym_space(v)
        return spaces.Dict(spec)
    elif isinstance(spec, specs.BoundedArray):
        return spaces.Box(low=spec.minimum, high=spec.maximum, shape=spec.shape, dtype=spec.dtype)
    elif isinstance(spec, specs.Array):
        return spaces.Box(low=-float('inf'), high=float('inf'), shape=spec.shape, dtype=spec.dtype)
    else:
        raise NotImplementedError

class dmc2gym(core.Env):
    def __init__(self, domain_name: str, task_name: str, task_kwargs: Optional[Dict] = {}, environment_kwargs=None):
        assert 'random' in task_kwargs, 'please specify a seed, for deterministic behaviour'
        self._env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs=task_kwargs, environment_kwargs=environment_kwargs)
        self.action_space = dmc_spec2gym_space(self._env.action_spec())
        self.observation_space = dmc_spec2gym_space(self._env.observation_spec())

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action: np.ndarray) -> TimeStep:
        assert self.action_space.contains(action)
        time_step = self._env.step(action)
        reward = time_step.reward or 0
        done = time_step.last()
        obs = time_step.observation
        info = {}
        if done and time_step.discount == 1.0:
            info['TimeLimit.truncated'] = True
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        return time_step.observation

    def render(self,
               mode='rgb_array',
               height: int = 84,
               width: int = 84,
               camera_id: int = 0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        _patch_dm_control_egl_py312()
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)

def _make_env_dmc(env_name: str, seed: int) -> gym.Env:
    domain_name, task_name = env_name.split("-")
    env = dmc2gym(domain_name=domain_name, task_name=task_name, task_kwargs={"random": seed})
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)
    from gym.wrappers import RescaleAction
    env = RescaleAction(env, -1.0, 1.0)
    env = SinglePrecision(env)
    if hasattr(env, "seed"):
        env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

class make_env_dmc(gym.Env):
    def __init__(self, env_name: str, seed: int, num_envs: int, max_t=1000):
        env_fns = [lambda i=i: _make_env_dmc(env_name, seed + i) for i in range(num_envs)]
        self.envs = [env_fn() for env_fn in env_fns]
        self.env_name = env_name
        self.max_t = max_t
        self.num_seeds = len(self.envs)
        self.action_space = spaces.Box(low=self.envs[0].action_space.low[None].repeat(len(self.envs), axis=0),
                                       high=self.envs[0].action_space.high[None].repeat(len(self.envs), axis=0),
                                       shape=(len(self.envs), self.envs[0].action_space.shape[0]),
                                       dtype=self.envs[0].action_space.dtype)
        self.observation_space = spaces.Box(low=self.envs[0].observation_space.low[None].repeat(len(self.envs), axis=0),
                                            high=self.envs[0].observation_space.high[None].repeat(len(self.envs), axis=0),
                                            shape=(len(self.envs), self.envs[0].observation_space.shape[0]),
                                            dtype=self.envs[0].observation_space.dtype)

    def _reset_idx(self, idx):
        return self.envs[idx].reset()

    def reset_where_done(self, observations, terms, truns):
        resets = np.zeros((terms.shape))
        for j, (term, trun) in enumerate(zip(terms, truns)):
            if (term == True) or (trun == True):
                observations[j], terms[j], truns[j] = self._reset_idx(j), False, False
                resets[j] = 1
        return observations, terms, truns, resets

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

    def reset(self):
        obs = []
        for env in self.envs:
            obs.append(env.reset())
        return np.stack(obs)

    def step(self, actions):
        obs, rews, terms, truns = [], [], [], []
        for env, action in zip(self.envs, actions):
            ob, reward, done, info = env.step(action)
            obs.append(ob)
            rews.append(reward)
            terms.append(False)
            trun = True if 'TimeLimit.truncated' in info else False
            truns.append(trun)
        return np.stack(obs), np.stack(rews), np.stack(terms), np.stack(truns), None

    # Suite adapters: For DMC, do not change composition; model predicts full next obs.
    def observation_to_model_prediction(self, observations):
        return observations

    def model_prediction_to_observation(self, cur_observations, pred_current_block):
        return pred_current_block

    def _video_fps(self):
        env = self.envs[0]
        for _ in range(12):
            try:
                if hasattr(env, "control_timestep"):
                    dt = float(env.control_timestep())
                    if dt > 0:
                        return max(1, int(round(1.0 / dt)))
            except Exception:
                pass
            try:
                if hasattr(env, "dt"):
                    dt = float(env.dt)
                    if dt > 0:
                        return max(1, int(round(1.0 / dt)))
            except Exception:
                pass
            next_env = getattr(env, "env", None)
            if next_env is None:
                break
            env = next_env
        return 30

    def evaluate(self, agent, num_episodes=5, save_video=False, step=None):
        num_seeds = self.num_seeds
        returns_eval = []
        for episode in range(num_episodes):
            observations = self.reset()

            returns = np.zeros(num_seeds)
            frames = []
            for i in range(self.max_t): # CHANGE?
                if save_video and episode == 0:
                    try:
                        frames.append(self.envs[0].render(mode='rgb_array', height=256, width=256))
                    except Exception:
                        pass
                
                actions = agent.sample_actions(observations, temperature=0.0)
                next_observations, rewards, terms, truns, goals = self.step(actions)

                returns += rewards
                observations = next_observations            
                
            if save_video and episode == 0 and len(frames) > 0:
                import wandb
                try:
                    video_array = np.array(frames)
                    if len(video_array.shape) == 4 and video_array.shape[-1] == 3:
                        video_array = np.transpose(video_array, (0, 3, 1, 2))
                    # Fallback environment name handling
                    env_name = getattr(self, 'env_name', 'dmc_env')
                    wandb.log({f"eval_video/{env_name}": wandb.Video(video_array, fps=self._video_fps(), format="mp4")}, step=step)
                except Exception as e:
                    print(f"Failed to log video to wandb: {e}")
                    
            returns_eval.append(returns)
        return {'return': np.array(returns_eval).mean(axis=0)}