"""I-24 subnetwork example."""
import os

import numpy as np

from flow.controllers.car_following_models import IDMController

#Specific to using to control adverarial vehicles:
from flow.controllers.car_following_adversarial import ACC_Switched_Controller_Attacked
from flow.controllers.lane_change_controllers import StaticLaneChanger
from flow.controllers.routing_controllers import i24_adversarial_router
from flow.controllers.routing_controllers import I24Router

# from flow.controllers.lane_change_controllers import AILaneChangeController
# from flow.controllers.lane_change_controllers import I24_routing_LC_controller
# from flow.controllers.routing_controllers import I210Router

# For flow:
from flow.core.params import SumoParams
from flow.core.params import EnvParams
from flow.core.params import NetParams
from flow.core.params import SumoLaneChangeParams
from flow.core.params import VehicleParams
from flow.core.params import InitialConfig
from flow.core.params import InFlows

from flow.core.params import SumoCarFollowingParams

import flow.config as config
from flow.envs import TestEnv

#Needed for i24 network:
from flow.networks.I24_Subnetwork_test_merge import I24SubNetwork
from flow.networks.I24_Subnetwork_test_merge import EDGES_DISTRIBUTION


acc_penetration=.01
inflow=2400
attack_magnitude=-.05
attack_duration=5.0
emission_path = '/Users/vanderbilt/Desktop/Research_2020/Traffic_Attack/flow/examples/adversarial_sims/'

SIM_LENGTH = 800 #simulation length in seconds
sim_step = .1 #Simulation step size

horizon = int(np.floor(SIM_LENGTH/sim_step)) #Number of simulation steps

WARMUP_STEPS = 1000 #Attack vehicles don't attack before this # of steps

BASELINE_INFLOW_PER_LANE = inflow #Per lane flow rate in veh/hr

inflow_speed = 25.5

ON_RAMP_FLOW = 1000

highway_start_edge = 'Eastbound_3'

ACC_PENETRATION_RATE = acc_penetration

HUMAN_INFLOW = (1-ACC_PENETRATION_RATE)*BASELINE_INFLOW_PER_LANE
ACC_INFLOW = (ACC_PENETRATION_RATE)*BASELINE_INFLOW_PER_LANE


##################################
#ATTACK VEHICLE PARAMETERS:
##################################

vehicles = VehicleParams()

inflow = InFlows()

attack_magnitude = -np.abs(attack_magnitude)

attack_duration = attack_duration
attack_magnitude = attack_magnitude
adversary_accel_controller = (ACC_Switched_Controller_Attacked,{
    'warmup_steps':WARMUP_STEPS,
    'Total_Attack_Duration':attack_duration,
    'attack_decel_rate':attack_magnitude})
adversarial_router = (i24_adversarial_router,{})

##################################
#DRIVER TYPES AND INFLOWS:
##################################
lane_list = ['0','1','2','3']


# ACC params and inflows:
vehicles.add(
    "attacker",
    num_vehicles=0,
    color="blue",
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode=0,
    ),
    # this is only right of way on
    car_following_params=SumoCarFollowingParams(
        speed_mode=0  # right of way at intersections + obey limits on deceleration
    ),
    acceleration_controller=adversary_accel_controller,
    lane_change_controller=(StaticLaneChanger,{}),
    routing_controller=adversarial_router, #This breaks everything
)
for i,lane in enumerate(lane_list):
    inflow.add(
        veh_type="attacker",
        edge=highway_start_edge,
        vehs_per_hour=ACC_INFLOW,
        departLane=lane,
        departSpeed=inflow_speed)

#handles when vehicles wait too long to try and merge and get stuck on merge:
human_routing_controller = (I24Router,{'position_to_switch_routes':5})

#Human params and inflows (main line and on-ramp)
vehicles.add(
    "human_main",
    num_vehicles=0,
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode=597,
    ),
    # this is only right of way on
    car_following_params=SumoCarFollowingParams(
        min_gap=0.5,
        speed_mode=12  # right of way at intersections + obey limits on deceleration
    ),
    routing_controller=human_routing_controller,

)

vehicles.add(
    "human_on_ramp",
    num_vehicles=0,
    # color="red",
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode=597,
    ),
    # this is only right of way on
    car_following_params=SumoCarFollowingParams(
        min_gap=0.5,
        speed_mode=12  # right of way at intersections + obey limits on deceleration
    ),
    routing_controller=human_routing_controller,
)

for i,lane in enumerate(lane_list):
    inflow.add(
        veh_type="human_main",
        edge=highway_start_edge,
        vehs_per_hour=HUMAN_INFLOW,
        departLane=lane,
        departSpeed=inflow_speed)

inflow.add(
    veh_type="human_on_ramp",
    edge='Eastbound_On_1',
    vehs_per_hour=ON_RAMP_FLOW,
    departLane='random',
    departSpeed=20)

##################################
#INITIALIZE FLOW PARAMETERS DICT:
##################################


NET_TEMPLATE = os.path.join(
        config.PROJECT_PATH,
        "examples/exp_configs/templates/sumo/i24_subnetwork_fix_merges.net.xml")

flow_params = dict(
    # name of the experiment
    exp_tag='I-24_subnetwork',

    # name of the flow environment the experiment is running on
    env_name=TestEnv,

    # name of the network class the experiment is running on
    network=I24SubNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # simulation-related parameters
    sim=SumoParams(
        sim_step=sim_step,
        render=False,
        color_by_speed=False,
        use_ballistic=True,
        emission_path=emission_path
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=horizon,
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        template=NET_TEMPLATE,
        additional_params={"on_ramp": False,'ghost_edge':False}
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        edges_distribution=EDGES_DISTRIBUTION,
    ),
)