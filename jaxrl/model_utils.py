from jaxrl.replay_buffer import Batch
import numpy as np
import jax
    
def get_coef(r_all, s_all):
    r = r_all.mean(axis=[0, 3]).squeeze()
    s = s_all.mean(axis=[0, 3])
    r_std = r_all.std(axis=[0, 3])
    s_std = s_all.std(axis=[0, 3])
    std = (r_std + s_std).mean(axis=-1)
    return r, s, 1./(std + 1)

def rollout_model(agent, observations, dones, masks, H=1):
    obs = observations
    actions = agent.sample_actions(obs, temperature=1.0)
    r1_all, s1_all = agent.model_predict_many(obs, actions)
    r1, s1, coef = get_coef(r1_all, s1_all)

    batches = []
    batches.append(Batch(
        observations=obs,
        actions=actions,
        next_observations=s1,
        rewards=r1,
        dones=dones,
        masks=masks,
        coefs=coef
    ))

    cur_s = s1
    cur_nxt_act = agent.sample_actions(cur_s, temperature=1.0)
    for i in range(H - 1):
        r_all, s_all = agent.model_predict_many(cur_s, cur_nxt_act)
        r, s, coef = get_coef(r_all, s_all)
                
        nxt_act = agent.sample_actions(s, temperature=1.0)
        batches.append(Batch(
            observations=cur_s,
            actions=cur_nxt_act,
            next_observations=s,
            rewards=r,
            dones=dones,
            masks=masks,
            coefs=coef
        ))
        cur_s = s
        cur_nxt_act = nxt_act

    return batches


def rollout(agent, replay_buffer, model_replay_buffer, H=1, bs=20000, n=5, num_consider=100000):
    fake_batches = []
    
    for j in range(n):
        if num_consider == -1:
            batch = replay_buffer.sample_parallel(bs)
        else:
            batch = replay_buffer.sample_parallel_last(num_consider, bs)
        cur_batches = rollout_model(agent, batch.observations, batch.dones, batch.masks, H=H)
        fake_batches.extend(cur_batches)
    
    n_rollouts = fake_batches[0].observations.shape[1]

    for j in range(len(fake_batches)):
        start_pos = n_rollouts * j
        model_replay_buffer.observations[:, start_pos:start_pos + n_rollouts] = fake_batches[j].observations
        model_replay_buffer.actions[:, start_pos:start_pos + n_rollouts] = fake_batches[j].actions
        model_replay_buffer.rewards[:, start_pos:start_pos + n_rollouts] = fake_batches[j].rewards
        model_replay_buffer.masks[:, start_pos:start_pos + n_rollouts] = fake_batches[j].masks
        model_replay_buffer.dones_float[:, start_pos:start_pos + n_rollouts] = fake_batches[j].dones
        model_replay_buffer.next_observations[:, start_pos:start_pos + n_rollouts] = fake_batches[j].next_observations
        model_replay_buffer.coefs[:, start_pos:start_pos + n_rollouts] = fake_batches[j].coefs

    model_replay_buffer.size = n_rollouts * len(fake_batches)
    