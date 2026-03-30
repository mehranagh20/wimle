import numpy as np
import wandb
import logging
import tensorflow_probability.substrates.numpy as tfp

def compute_iqm(scores):
    """
    Compute Interquartile Mean (IQM) - mean of middle 50% of scores.
    Uses the same method as rliable.metrics.aggregate_iqm.
    
    Args:
        scores: numpy array of scores
    
    Returns:
        float: IQM value
    """
    flattened_scores = np.asarray(scores).flatten()
    
    # Calculate 25th and 75th percentiles using "nearest" method (same as rliable)
    q25 = np.percentile(flattened_scores, 25, method='nearest')
    q75 = np.percentile(flattened_scores, 75, method='nearest')
    
    # Keep only scores in the interquartile range [Q1, Q3]
    middle_50_percent = flattened_scores[
        (flattened_scores >= q25) & (flattened_scores <= q75)
    ]
    
    # Return the mean of middle 50%
    return np.mean(middle_50_percent)

def aggregate_info_dicts(dicts):
    """Aggregate a list of info dictionaries by concatenating arrays for each key."""
    agg = {}
    if not dicts:
        return agg
    keys = set()
    for d in dicts:
        keys.update(d.keys())
    for k in keys:
        values = []
        for d in dicts:
            if k in d:
                v = d[k]
                v_arr = np.asarray(v)
                if v_arr.ndim == 0:
                    v_arr = v_arr[None]
                values.append(v_arr)
        if values:
            agg[k] = np.concatenate(values, axis=0)
    return agg


def mute_warning():
    tfp.distributions.TransformedDistribution(tfp.distributions.Normal(0.0, 1.0), tfp.bijectors.Identity())
    logger = logging.getLogger('root')
    class CheckTypesFilter(logging.Filter):
        def filter(self, record):
            return 'check_types' not in record.getMessage()
    logger.addFilter(CheckTypesFilter())

def log_to_wandb_if_time_to(step, infos, eval_interval, suffix: str = ''):
    if step % eval_interval == 0:
        dict_to_log = {'timestep': step}
        for info_key in infos:
            values = infos[info_key]
            # Log individual seed values
            for seed, value in enumerate(values):
                dict_to_log[f'seed{seed}/{info_key}{suffix}'] = value
            
            # Add mean and std if values are numerical
            try:
                values_array = np.array(values)
                if np.issubdtype(values_array.dtype, np.number):
                    dict_to_log[f'mean/{info_key}{suffix}'] = float(np.mean(values_array))
                    dict_to_log[f'std/{info_key}{suffix}'] = float(np.std(values_array))
                    dict_to_log[f'min/{info_key}{suffix}'] = float(np.min(values_array))
                    dict_to_log[f'max/{info_key}{suffix}'] = float(np.max(values_array))
                    
                    # Add IQM calculation only for 'return' key
                    if info_key == 'return' or info_key == 'goal':
                        try:
                            iqm_value = compute_iqm(values_array)
                            dict_to_log[f'iqm/{info_key}{suffix}'] = float(iqm_value)
                        except Exception:
                            # If IQM calculation fails, continue without it
                            pass
            except:
                pass  # Skip mean/std for non-numerical values
                
        wandb.log(dict_to_log, step=step)


def capture_seed_frames(envs, frames_by_seed, render_fn):
    """Capture one frame per seed env and append into frames_by_seed."""
    for seed_idx, env in enumerate(envs):
        try:
            frame = render_fn(env)
            if frame is not None:
                frames_by_seed[seed_idx].append(frame)
        except Exception:
            pass


def log_seed_videos_to_wandb(frames_by_seed, env_name, fps, step, prefix='eval_video'):
    """Log one video per seed to W&B under <prefix>/<env_name>/seed<idx>."""
    videos_to_log = {}
    for seed_idx, seed_frames in enumerate(frames_by_seed):
        if len(seed_frames) == 0:
            continue
        video_array = np.array(seed_frames)
        if len(video_array.shape) == 4 and video_array.shape[-1] == 3:
            video_array = np.transpose(video_array, (0, 3, 1, 2))
        videos_to_log[f"{prefix}/{env_name}/seed{seed_idx}"] = wandb.Video(
            video_array, fps=fps, format="mp4"
        )
    if len(videos_to_log) > 0:
        wandb.log(videos_to_log, step=step)


def evaluate_agents_if_time(i, agents, eval_envs, eval_interval, eval_episodes, save_video=False):
    """Evaluate multiple agents across multiple environments if it's time to evaluate."""
    if i % eval_interval != 0:
        return
    eval_dicts = []
    for agent, eval_env in zip(agents, eval_envs):
        stats = eval_env.evaluate(agent, num_episodes=eval_episodes, save_video=save_video, step=i)
        eval_dicts.append(stats)
    combined = aggregate_info_dicts(eval_dicts)
    log_to_wandb_if_time_to(i, combined, eval_interval, suffix='_eval')
    
    # Log the return array to wandb as np_return
    if 'return' in combined:
        wandb.log({'np_return': combined['return']}, step=i)
    
    if 'goal' in combined:
        wandb.log({'np_goal': combined['goal']}, step=i)


def make_env(benchmark, env_name, seed, num_envs):
    if benchmark == 'dmc':
        from jaxrl.envs.dmc_gym import make_env_dmc
        return make_env_dmc(env_name, seed=seed, num_envs=num_envs)
    if benchmark == 'hb':
        from jaxrl.envs.humanoid_bench_gym import make_env_hb
        return make_env_hb(env_name, seed=seed, num_envs=num_envs)
    if benchmark == 'mw':
        from jaxrl.envs.metaworld_gym import make_env_mw
        return make_env_mw(env_name, seed=seed, num_envs=num_envs)
    if benchmark == 'myo':
        from jaxrl.envs.myosuite_gym import make_env_myo
        return make_env_myo(env_name, seed=seed, num_envs=num_envs)

        