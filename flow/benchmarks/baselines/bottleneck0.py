"""Evaluates the baseline performance of bottleneck0 without RL control.

Baseline is no AVs.
"""

import numpy as np
from flow.core.experiment import SumoExperiment
from flow.core.params import InitialConfig
from flow.core.vehicles import Vehicles
from flow.core.traffic_lights import TrafficLights
from flow.controllers import ContinuousRouter
from flow.benchmarks.bottleneck0 import flow_params, SCALING


def bottleneck0_baseline(num_runs, render=True):
    """Run script for the bottleneck0 baseline.

    Parameters
    ----------
        num_runs : int
            number of rollouts the performance of the environment is evaluated
            over
        render : bool, optional
            specifies whether to use sumo's gui during execution

    Returns
    -------
        SumoExperiment
            class needed to run simulations
    """
    exp_tag = flow_params["exp_tag"]
    sumo_params = flow_params["sumo"]
    env_params = flow_params["env"]
    net_params = flow_params["net"]
    initial_config = flow_params.get("initial", InitialConfig())
    traffic_lights = flow_params.get("tls", TrafficLights())

    # remove autonomous vehicles
    vehicles = Vehicles()
    vehicles.add(veh_id="human",
                 speed_mode=9,
                 routing_controller=(ContinuousRouter, {}),
                 lane_change_mode=0,
                 num_vehicles=1 * SCALING)

    # modify the rendering to match what is requested
    sumo_params.render = render

    # import the scenario class
    module = __import__("flow.scenarios", fromlist=[flow_params["scenario"]])
    scenario_class = getattr(module, flow_params["scenario"])

    # create the scenario object
    scenario = scenario_class(
        name=exp_tag,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=traffic_lights
    )

    # import the environment class
    module = __import__("flow.envs", fromlist=[flow_params["env_name"]])
    env_class = getattr(module, flow_params["env_name"])

    # create the environment object
    env = env_class(env_params, sumo_params, scenario)

    exp = SumoExperiment(env, scenario)

    results = exp.run(num_runs, env_params.horizon)
    return np.mean(results["returns"]), np.std(results["returns"])


if __name__ == "__main__":
    runs = 2  # number of simulations to average over
    mean, std = bottleneck0_baseline(num_runs=runs, render=True)

    print('---------')
    print('The average outflow, std. deviation over 500 seconds '
          'across {} runs is {}, {}'.format(runs, mean, std))
