import argparse
import gym
import numpy as np
import os
import sys
import time

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env

from flow.core.util import emission_to_csv
from flow.utils.registry import make_create_env
from flow.utils.rllib import get_flow_params
from flow.utils.rllib import get_rllib_config
from flow.utils.rllib import get_rllib_pkl


# This looks like generic code for running a simulation given a generic agent:


    result_dir = args.result_dir if args.result_dir[-1] != '/' \
    else args.result_dir[:-1]

    config = get_rllib_config(result_dir)


    create_env, env_name = make_create_env(params=flow_params, version=0)
    register_env(env_name, create_env)


    # check if we have a multiagent environment but in a
    # backwards compatible way
    if config.get('multiagent', {}).get('policies', None):
        multiagent = True
        pkl = get_rllib_pkl(result_dir)
        config['multiagent'] = pkl['multiagent']
    else:
        multiagent = False

    # Run on only one cpu for rendering purposes
    config['num_workers'] = 0

    flow_params = get_flow_params(config)




    # create the agent that will be used to compute the actions
    agent = agent_cls(env=env_name, config=config)
    checkpoint = result_dir + '/checkpoint_' + args.checkpoint_num
    checkpoint = checkpoint + '/checkpoint-' + args.checkpoint_num
    agent.restore(checkpoint)


    for _ in range(env_params.horizon):
        vehicles = env.unwrapped.k.vehicle
        speeds = vehicles.get_speed(vehicles.get_ids())

        # only include non-empty speeds
        if speeds:
            vel.append(np.mean(speeds))

        if multiagent:
            action = {}
            for agent_id in state.keys():
                if use_lstm:
                    action[agent_id], state_init[agent_id], logits = \
                        agent.compute_action(
                        state[agent_id], state=state_init[agent_id],
                        policy_id=policy_map_fn(agent_id))
                else:
                    action[agent_id] = agent.compute_action(
                        state[agent_id], policy_id=policy_map_fn(agent_id))
        else:
            action = agent.compute_action(state)
        state, reward, done, _ = env.step(action)
        if multiagent:
            for actor, rew in reward.items():
                ret[policy_map_fn(actor)][0] += rew
        else:
            ret += reward
        if multiagent and done['__all__']:
            break
        if not multiagent and done:
            break