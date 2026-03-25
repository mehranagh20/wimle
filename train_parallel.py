import sys
import os
import time

os.environ['MUJOCO_GL'] = 'egl'

import random
import numpy as np
import tqdm
from absl import app, flags
import jax.numpy as jnp
import wandb

from jaxrl.wimle.wimle_learner import WIMLELearner
from jaxrl.replay_buffer import ParallelReplayBuffer
from jaxrl.utils import (
    mute_warning,
    make_env,
    log_to_wandb_if_time_to,
    aggregate_info_dicts,
    evaluate_agents_if_time
)
from jaxrl.model_utils import rollout
from hps import define_train_parallel_flags

define_train_parallel_flags()
FLAGS = flags.FLAGS


def main(_):
    save_dir = f'{FLAGS.save_dir}/results/{FLAGS.run_name}/'

    if FLAGS.model_updates_per_step is None:
        FLAGS.model_updates_per_step = FLAGS.updates_per_step

    wandb.init(
        config=FLAGS,
        project=FLAGS.wandb_project,
        group=f'{FLAGS.env_name}',
        name=FLAGS.run_name,
        mode=FLAGS.wandb_mode
    )
    wandb.run.log_code()

    os.makedirs(save_dir, exist_ok=True)
    env = make_env(FLAGS.benchmark, FLAGS.env_name, FLAGS.seed, num_envs=FLAGS.num_seeds)
    eval_env = make_env(FLAGS.benchmark, FLAGS.env_name, FLAGS.seed + 42, num_envs=FLAGS.num_seeds)
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    mute_warning()

    agent = WIMLELearner(
        FLAGS.seed,
        env,
        env.observation_space.sample()[0, np.newaxis],
        env.action_space.sample()[0, np.newaxis],
        num_seeds=FLAGS.num_seeds,
        lr=FLAGS.lr,
        world_model_lr=FLAGS.model_lr,
        discount=FLAGS.discount,
        tau=FLAGS.tau,
        init_temperature=FLAGS.init_temperature,
        target_entropy=None,
        n_quantiles=FLAGS.n_quantiles,
        n_models=FLAGS.num_models,
        model_depth=FLAGS.model_depth,
        model_hidden_dims=FLAGS.model_hidden_dims,
        latent_size=FLAGS.model_latent_size,
        num_latent_codes=FLAGS.model_num_latent_codes,
    )

    replay_buffer = ParallelReplayBuffer(env.observation_space, env.action_space.shape[-1], FLAGS.replay_buffer_size, num_seeds=FLAGS.num_seeds)
    model_replay_buffer = ParallelReplayBuffer(env.observation_space, env.action_space.shape[-1], capacity=2000000, num_seeds=FLAGS.num_seeds)

    observations = env.reset()
    for i in tqdm.tqdm(range(0, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm):
        actions = env.action_space.sample() if i < FLAGS.start_training else agent.sample_actions(observations, temperature=1.0)
        next_observations, rewards, terms, truns, _ = env.step(actions)
        masks = env.generate_masks(terms, truns)
        replay_buffer.insert(observations, actions, rewards, masks, truns, next_observations, jnp.ones(rewards.shape))
        observations = next_observations
        observations, terms, truns, _ = env.reset_where_done(observations, terms, truns)

        if i % FLAGS.model_train_freq == 0 and FLAGS.use_model:
            agent.train_world_model_ensemble(replay_buffer, FLAGS.model_batch_size, FLAGS.model_num_updates, num_consider=FLAGS.num_consider)
            rollout(
                agent,
                replay_buffer,
                model_replay_buffer,
                H=FLAGS.model_H,
                bs=FLAGS.model_batch_size,
                n=FLAGS.model_num_rollouts,
                num_consider=FLAGS.num_consider,
            )

        num_updates = FLAGS.updates_per_step
        batches = replay_buffer.sample_parallel_multibatch(FLAGS.batch_size, FLAGS.updates_per_step)
        if FLAGS.use_model:
            model_batches = model_replay_buffer.sample_parallel_multibatch(FLAGS.batch_size, FLAGS.model_updates_per_step)
            batches = replay_buffer.concat_batches(batches, model_batches)
            num_updates += FLAGS.model_updates_per_step
        infos = agent.update(batches, num_updates, i)

        if i % FLAGS.save_freq == 0:
            policy_save_path = os.path.join(save_dir, f'policy_step_{i}_instance_0.pkl')
            agent.actor.save(policy_save_path)
            print(f'Saved policy at step {i}')

        # Aggregate/log and evaluate only at intervals
        if i % FLAGS.eval_interval == 0:
            aggregated_infos = aggregate_info_dicts([infos])
            log_to_wandb_if_time_to(i, aggregated_infos, FLAGS.eval_interval)
            if i > 0:
                evaluate_agents_if_time(i, [agent], [eval_env], FLAGS.eval_interval, FLAGS.eval_episodes, save_video=getattr(FLAGS, 'save_video', False))

    wandb.finish()


if __name__ == '__main__':
    app.run(main)
