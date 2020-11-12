import os

import numpy as np

from flow.controllers.car_following_models import IDMController

#Specific to using to control adverarial vehicles:
from flow.controllers.car_following_adversarial import ACC_Switched_Controller_Attacked
from flow.controllers.lane_change_controllers import StaticLaneChanger
from flow.controllers.routing_controllers import i24_adversarial_router


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

#For running a simulation:
from flow.core.experiment import Experiment

import ray
ray.init()

def get_flow_params(attack_duration,attack_magnitude,attack_inflow,emission_path):
	horizon = 8000 #number of simulation steps
	sim_step = .1 #Simulation step size

	WARMUP_STEPS = 1000

	HUMAN_INFLOW = 2050 #Per lane flow rate in veh/hr
	inflow_speed = 25.5

	ON_RAMP_FLOW = 500

	highway_start_edge = 'Eastbound_3'

	lane_list = ['0','1','2','3']
	HUMAN_INFLOW_RATES_MAIN = [HUMAN_INFLOW,HUMAN_INFLOW,HUMAN_INFLOW*.8,HUMAN_INFLOW*.5]


	vehicles = VehicleParams()

	inflow = InFlows()

	# Parameters with respect to adversarial attacks:

	want_adversaries = True

	adversary_inflows = attack_inflow

	adversary_accel_controller = (ACC_Switched_Controller_Attacked,{
		'warmup_steps':WARMUP_STEPS,
		'Total_Attack_Duration':attack_duration,
		'attack_decel_rate':attack_magnitude}) #Default values

	adversarial_router = (i24_adversarial_router,{})


	if(want_adversaries):   

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

	    inflow.add(
	        veh_type="attacker",
	        edge=highway_start_edge,
	        vehs_per_hour=adversary_inflows,
	        departLane='random',
	        departSpeed=inflow_speed)


	human_accel_controller = (IDMController, {
	        "a": 1.3,
	        "b": 2.0,
	        "noise": 0.3,
	        "v0": 27.0,
	        "display_warnings": False,
	        "fail_safe": ['obey_speed_limit'],
	    })

	want_IDM = False

	# Decide whether to use IDM or the default SUMO model:

	if(want_IDM):
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
	        acceleration_controller=human_accel_controller,
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
	        acceleration_controller=human_accel_controller,
	    )
	else:
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
	    )

	for i,lane in enumerate(lane_list):
	    inflow.add(
	        veh_type="human_main",
	        edge=highway_start_edge,
	        vehs_per_hour=HUMAN_INFLOW_RATES_MAIN[i],
	        departLane=lane,
	        departSpeed=inflow_speed)

	inflow.add(
	    veh_type="human_on_ramp",
	    edge='Eastbound_On_1',
	    vehs_per_hour=ON_RAMP_FLOW,
	    departLane='random',
	    departSpeed=20)


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
	        emission_path=emission_path,
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

	return flow_params

def run_attack_sim(attack_duration,attack_magnitude,attack_inflow,emission_path,rename_file):
	
	flow_params = get_flow_params(attack_duration,attack_magnitude,attack_inflow,emission_path)

	exp = Experiment(flow_params)

	exp.run(num_runs=1,convert_to_csv=True)

	if(rename_file):
		files = os.listdir(emission_path)
		for file in files:
			if(file[:3] != 'Dur'):
				csv_path = os.path.join(emission_path,file)

		file_name_no_version = 'Dur_'+str(attack_duration)+'_Mag_'+str(attack_magnitude)+'_Inflow_'+str(attack_inflow)

		file_version = 1

		for file in files:
			if(file_name_no_version in file):
				file_version += 1

		file_path = os.path.join(emission_path,file_name_no_version+'_ver_'+str(file_version)+'.csv')


		os.rename(csv_path,file_path)

@ray.remote
def run_attack_sim_ray(attack_duration,attack_magnitude,attack_inflow,emission_path,rename_file):
	
	flow_params = get_flow_params(attack_duration,attack_magnitude,attack_inflow,emission_path)

	exp = Experiment(flow_params)

	exp.run(num_runs=1,convert_to_csv=True)

	if(rename_file):
		files = os.listdir(emission_path)
		for file in files:
			if(file[:3] != 'Dur'):
				csv_path = os.path.join(emission_path,file)

		file_name_no_version = 'Dur_'+str(attack_duration)+'_Mag_'+str(attack_magnitude)+'_Inflow_'+str(attack_inflow)

		file_version = 1

		for file in files:
			if(file_name_no_version in file):
				file_version += 1

		file_path = os.path.join(emission_path,file_name_no_version+'_ver_'+str(file_version)+'.csv')


		os.rename(csv_path,file_path)


def inflow_iter(attack_duration,attack_magnitude,attack_inflows,emission_path,rename_file,want_parallel):

	if(want_parallel):
		for attack_inflow in attack_inflows:
			run_attack_sim_ray(attack_duration,attack_magnitude,attack_inflow,emission_path,rename_file)
	else:
		for attack_inflow in attack_inflows:
			run_attack_sim(attack_duration,attack_magnitude,attack_inflow,emission_path,rename_file)
	
	print('Simulations finished.')
















