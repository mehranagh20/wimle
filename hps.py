from absl import flags


_FLAGS_DEFINED = False


def define_train_parallel_flags() -> None:
    global _FLAGS_DEFINED
    if _FLAGS_DEFINED:
        return

    # Run
    flags.DEFINE_string('save_dir', './tmp/', 'Logging and checkpoint directory.')
    flags.DEFINE_integer('seed', 0, 'Random seed.')
    flags.DEFINE_integer('num_seeds', 4, 'Number of parallel environment seeds.')
    flags.DEFINE_boolean('tqdm', True, 'Show tqdm progress bar.')
    flags.DEFINE_boolean('save_video', True, 'Save evaluation video to wandb.')

    # Environment
    flags.DEFINE_string('benchmark', 'dmc', 'Benchmark suite (e.g. dmc, gym).')
    flags.DEFINE_string('env_name', 'cheetah-run', 'Environment name.')

    # Training loop
    flags.DEFINE_integer('max_steps', int(1000000), 'Total training steps.')
    flags.DEFINE_integer('start_training', int(2500), 'Steps of random data before training.')
    flags.DEFINE_integer('batch_size', 128, 'Mini-batch size.')
    flags.DEFINE_integer('replay_buffer_size', int(1000000), 'Replay buffer capacity.')
    flags.DEFINE_integer('updates_per_step', 10, 'Policy updates per env step.')
    flags.DEFINE_integer('model_updates_per_step', 10, 'Model updates per step (default: same as updates_per_step).')
    flags.DEFINE_integer('eval_interval', 10000, 'Evaluate every N steps.')
    flags.DEFINE_integer('eval_episodes', 5, 'Episodes per evaluation.')
    flags.DEFINE_integer('save_freq', 500000000, 'Save policy every N steps.')

    # WandB
    flags.DEFINE_string('wandb_project', 'WIMLE', 'Wandb project name.')
    flags.DEFINE_string('run_name', 'WIMLE', 'Wandb run name.')
    flags.DEFINE_string('wandb_mode', 'online', 'Wandb mode (online/offline).')

    # Agent (actor / critic)
    flags.DEFINE_float('lr', 3e-4, 'Learning rate (actor, critic, temperature).')
    flags.DEFINE_float('discount', 0.99, 'Discount factor.')
    flags.DEFINE_float('tau', 0.005, 'Soft target update coefficient.')
    flags.DEFINE_float('init_temperature', 1.0, 'Initial temperature for SAC.')
    flags.DEFINE_integer('n_quantiles', 100, 'Number of quantiles for critic.')

    # World model
    flags.DEFINE_integer('use_model', 1, 'Use world model (1) or not (0).')
    flags.DEFINE_float('model_lr', 1e-3, 'World model learning rate.')
    flags.DEFINE_integer('model_batch_size', 512, 'Model training batch size.')
    flags.DEFINE_integer('model_num_updates', 100, 'Model updates per train call.')
    flags.DEFINE_integer('model_num_rollouts', 200, 'Rollouts per step.')
    flags.DEFINE_integer('model_train_freq', 1000, 'Train model every N steps.')
    flags.DEFINE_integer('model_depth', 3, 'Model network depth.')
    flags.DEFINE_integer('model_hidden_dims', 512, 'Model hidden width.')
    flags.DEFINE_integer('model_latent_size', 4, 'Model latent dimension.')
    flags.DEFINE_integer('model_num_latent_codes', 4, 'Number of latent codes for nearest neighbour.')
    flags.DEFINE_integer('num_models', 7, 'Number of models in ensemble.')
    flags.DEFINE_integer('model_H', 4, 'Model rollout horizon.')
    flags.DEFINE_integer('num_consider', 100000, 'Recent samples to consider for model (-1 = all).')

    _FLAGS_DEFINED = True
