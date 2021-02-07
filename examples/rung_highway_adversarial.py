"""Example of an open multi-lane network with human-driven vehicles."""

from flow.controllers import IDMController
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, SumoLaneChangeParams
from flow.core.params import VehicleParams, InFlows
from flow.envs.ring.lane_change_accel import ADDITIONAL_ENV_PARAMS
from flow.networks.highway import HighwayNetwork, ADDITIONAL_NET_PARAMS
from flow.envs import LaneChangeAccelEnv


def get_flow_params(attack_duration,
    attack_magnitude,
    acc_penetration,
    inflow,
    emission_path,
    attack_penetration,
    want_render=False,
    display_attack_info=True)

    SIM_LENGTH = 1000

    sim_step = .1 #Simulation step size

    horizon = int(np.floor(SIM_LENGTH/sim_step)) #Number of simulation steps

    WARMUP_STEPS = 1000 #Attack vehicles don't attack before this # of steps

    NUM_VEHICLES = 100

    vehicle_length = 4.0

    vehicles = VehicleParams()

    ##################################
    # HUMAN VEHICLE PARAMETERS:
    ##################################

    human_accel = (IDMController,{'a':1.3,'b':1.4,'noise':.1})

    vehicles.add(
        veh_id="human",
        car_following_params=SumoCarFollowingParams(speed_mode=12,length=vehicle_length),
        acceleration_controller=human_accel,
        routing_controller=(ContinuousRouter, {}),)


    ##################################
    # ATTACK VEHICLE PARAMETERS:
    ##################################

    attack_magnitude = -np.abs(attack_magnitude)

    attack_duration = attack_duration
    attack_magnitude = attack_magnitude
    adversary_ACC_controller = (ACC_Switched_Controller_Attacked,{
        'warmup_steps':WARMUP_STEPS,
        'Total_Attack_Duration':attack_duration,
        'attack_decel_rate':attack_magnitude,
        'display_attack_info':display_attack_info})

    #Should never attack, just a regular ACC:
    benign_ACC_controller = (ACC_Switched_Controller_Attacked,{
        'warmup_steps':horizon, #Causes to never activate attacks
        'Total_Attack_Duration':0,
        'attack_decel_rate':0,
        'display_attack_info':display_attack_info})

    human_accel = (IDMController,{'a':1.3,'b':1.4,'noise':.1})

    vehicles = VehicleParams()

    vehicles.add(
        veh_id="human",
        car_following_params=SumoCarFollowingParams(speed_mode=12,length=vehicle_length),
        acceleration_controller=human_accel,
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=num_human_drivers)

    vehicles.add(
        veh_id="acc_benign",
        color="blue",
        car_following_params=SumoCarFollowingParams(speed_mode=12,length=vehicle_length),
        acceleration_controller=benign_ACC_controller,
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=num_benign_acc_drivers)

    vehicles.add(
        veh_id="acc_adversarial",
        color="red",
        car_following_params=SumoCarFollowingParams(speed_mode=12,length=vehicle_length),
        acceleration_controller=adversary_ACC_controller,
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=num_attack_acc_drivers)


    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    ##################################
    # CONSTRUCT INFLOWS:
    ##################################

    ACC_PENETRATION_RATE = acc_penetration

    HUMAN_INFLOW = (1-ACC_PENETRATION_RATE)*BASELINE_INFLOW_PER_LANE

    ACC_INFLOW = (ACC_PENETRATION_RATE)*BASELINE_INFLOW_PER_LANE

    ACC_ATTACK_INFLOW = (attack_penetration)*ACC_INFLOW

    ACC_BENIGN_INFLOW = (1-attack_penetration)*ACC_INFLOW

    inflow = InFlows()

    if(ACC_ATTACK_INFLOW > 0):
        inflow.add(
            veh_type="attacker_ACC",
            edge="highway_0",
            vehs_per_hour=ACC_ATTACK_INFLOW ,
            depart_lane=lane,
            depart_speed=inflow_speed)

    if(ACC_BENIGN_INFLOW > 0):
        inflow.add(
            veh_type="benign_ACC",
            edge="highway_0",
            vehs_per_hour=ACC_BENIGN_INFLOW ,
            depart_lane=lane,
            depart_speed=inflow_speed)






    flow_params = dict(
        # name of the experiment
        exp_tag='highway',

        # name of the flow environment the experiment is running on
        env_name=LaneChangeAccelEnv,

        # name of the network class the experiment is running on
        network=HighwayNetwork,

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            render=True,
            lateral_resolution=1.0,
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            horizon=1500,
            additional_params=ADDITIONAL_ENV_PARAMS.copy(),
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            inflows=inflow,
            additional_params=ADDITIONAL_NET_PARAMS.copy(),
        ),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon initialization/
        # reset (see flow.core.params.InitialConfig)
        initial=InitialConfig(
            spacing="uniform",
            shuffle=True,
        ),
    )
