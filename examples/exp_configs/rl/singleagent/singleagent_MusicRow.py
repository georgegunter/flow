"""Open merge example.

Trains a a small percentage of rl vehicles to dissipate shockwaves caused by
on-ramp merge to a single lane open highway network.
"""

import warnings

warnings.filterwarnings("ignore")


import numpy as np
import os

from flow.core.params import SumoParams, EnvParams, InitialConfig
from flow.core.params import NetParams, InFlows, SumoCarFollowingParams
from flow.networks.merge import ADDITIONAL_NET_PARAMS
from flow.core.params import VehicleParams
from flow.core.params import SumoParams
from flow.core.params import EnvParams
from flow.core.params import NetParams
from flow.core.params import SumoLaneChangeParams
from flow.core.params import VehicleParams
from flow.core.params import InitialConfig
from flow.core.params import InFlows
import flow.config as config
from flow.envs import TestEnv

from flow.core.experiment import Experiment

from flow.controllers import IDMController, RLController
from flow.controllers import SimLaneChangeController, ContinuousRouter
import flow.config as config


from flow.envs import MusicRow_POEnv
from flow.networks import MusicRowCorridor

# experiment number
# - 0: 10% RL penetration,  5 max controllable vehicles
# - 1: 25% RL penetration, 13 max controllable vehicles
# - 2: 33% RL penetration, 17 max controllable vehicles
EXP_NUM = 0

# time horizon of a single rollout
HORIZON = 300
# number of rollouts per training iteration
N_ROLLOUTS = 20
# number of parallel workers
N_CPUS = 1

FLOW_RATE = 1000

INFLOW_SPEED = 10.0


# percent of autonomous vehicles
RL_PENETRATION = [0.1, 0.25, 0.33][EXP_NUM]
# num_rl term (see ADDITIONAL_ENV_PARAMs)
NUM_RL = [5, 13, 17][EXP_NUM]



# we define an EDGES_DISTRIBUTION variable with the edges within the music row corridor:
# EDGES_DISTRIBUTION = list(np.loadtxt('music_row_16th_ave_edges.txt',dtype=str,comments=None))

# the above variable is added to initial_config
# new_initial_config = InitialConfig(
#     edges_distribution=EDGES_DISTRIBUTION
# )

NET_TEMPLATE = os.path.join(config.PROJECT_PATH,'examples/exp_configs/templates/sumo/music_row.net.xml')


# RL vehicles constitute 5% of the total number of vehicles
vehicles = VehicleParams()
vehicles.add(
    "human",
    num_vehicles=0,
    lane_change_controller=(SimLaneChangeController, {}),
    
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode=1621,
    ),
    
    routing_controller=(ContinuousRouter, {}),
    
    acceleration_controller=(IDMController, {
        "a": 1.3,
        "b": 2.0,
        "noise": 0.3,
    })
)


vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
    ),
    num_vehicles=0,
    color='red')

# Vehicles are introduced from both sides of merge, with RL vehicles entering
# from the highway portion as well
inflow = InFlows()

for lane in [0, 1]:
    inflow.add(
        veh_type="human",
        edge='19449685#0',
        vehs_per_hour=FLOW_RATE,
        depart_lane=lane,
        depart_speed=INFLOW_SPEED)


for lane in [0]:
    inflow.add(
        veh_type="rl",
        edge='19449685#0',
        vehs_per_hour=(RL_PENETRATION) * FLOW_RATE,
        depart_lane=lane,
        depart_speed=INFLOW_SPEED)


    flow_params = dict(
        # name of the experiment
        exp_tag='music_row_corridor',

        # name of the flow environment the experiment is running on
        env_name=MusicRow_POEnv,

        # name of the network class the experiment is running on
        network=MusicRowCorridor,

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            sim_step=0.2,
            render=False,
            restart_instance=True,
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            horizon=HORIZON,
            sims_per_step=5,
            warmup_steps=0,
            additional_params={
                "max_accel": 1.5,
                "max_decel": 1.5,
                "target_velocity": 20,
                "num_rl": NUM_RL,
            },
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            inflows=inflow,
            template=NET_TEMPLATE,
            additional_params={}
        ),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon initialization/
        # reset (see flow.core.params.InitialConfig)
        initial=InitialConfig(),
    )


# if __name__ == '__main__':

#     exp = Experiment(flow_params)

#     _ = exp.run(1)
