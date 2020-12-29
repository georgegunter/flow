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

#For running a simulation:
from flow.core.experiment import Experiment


# For procesing results:
from load_sim_results import get_sim_results
from load_sim_results import write_sim_results

# Ray:
import ray
ray.init(ignore_reinit_error=True)

def get_flow_params(attack_duration,attack_magnitude,acc_penetration,inflow,emission_path,attack_penetration=0.2):

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

	ACC_ATTACK_INFLOW = (attack_penetration)*ACC_INFLOW

	ACC_BENIGN_INFLOW = (1-attack_penetration)*ACC_INFLOW

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

	#Should never attack, so just a regular ACC:
	benign_ACC_controller = (ACC_Switched_Controller_Attacked,{
		'warmup_steps':horizon,
		'Total_Attack_Duration':0,
		'attack_decel_rate':0})

	##################################
	#DRIVER TYPES AND INFLOWS:
	##################################
	lane_list = ['0','1','2','3']

	# Attack ACC params and inflows:
	vehicles.add(
		"attacker_ACC",
		num_vehicles=0,
		color="red",
		lane_change_params=SumoLaneChangeParams(
			lane_change_mode=0,
		),
		# this is only right of way on
		car_following_params=SumoCarFollowingParams(
			speed_mode=0  # right of way at intersections + obey limits on deceleration
		),
		acceleration_controller=adversary_accel_controller,
		lane_change_controller=(StaticLaneChanger,{}),
		routing_controller=adversarial_router,
	)

	vehicles.add(
		"benign_ACC",
		num_vehicles=0,
		color="blue",
		lane_change_params=SumoLaneChangeParams(
			lane_change_mode=0,
		),
		# this is only right of way on
		car_following_params=SumoCarFollowingParams(
			speed_mode=0  # right of way at intersections + obey limits on deceleration
		),
		acceleration_controller=benign_ACC_controller,
		lane_change_controller=(StaticLaneChanger,{}),
		routing_controller=adversarial_router, #This breaks everything
	)


	for i,lane in enumerate(lane_list):
		if(ACC_ATTACK_INFLOW > 0):
			inflow.add(
				veh_type="attacker_ACC",
				edge=highway_start_edge,
				vehs_per_hour=ACC_ATTACK_INFLOW ,
				departLane=lane,
				departSpeed=inflow_speed)

		if(ACC_BENIGN_INFLOW > 0):
			inflow.add(
				veh_type="benign_ACC",
				edge=highway_start_edge,
				vehs_per_hour=ACC_BENIGN_INFLOW ,
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
			render=True,
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

	return flow_params

def run_attack_sim(attack_duration,attack_magnitude,acc_penetration,attack_penetration,inflow,emission_path,rename_file,get_results=True,delete_file=False):

	flow_params = get_flow_params(attack_duration,attack_magnitude,acc_penetration,inflow,emission_path,attack_penetration)

	exp = Experiment(flow_params)

	exp.run(num_runs=1,convert_to_csv=True)

	if(rename_file):
		files = os.listdir(emission_path)

		for file in files:
			if(file[:3] != 'Dur' and file[-3:] == 'csv'):
				csv_path = os.path.join(emission_path,file)

		file_name_no_version = 'Dur_'+str(attack_duration)+'_Mag_'+str(attack_magnitude)+'_Inflow_'+str(inflow)+'_ACCPenetration_'+str(acc_penetration)+'_AttackPenetration_'+str(attack_penetration)

		file_version = 1

		for file in files:
			if(file_name_no_version in file):
				file_version += 1

		file_name_with_version = file_name_no_version+'_ver_'+str(file_version)+'.csv'

		file_path = os.path.join(emission_path,file_name_with_version)


		os.rename(csv_path,file_path)

	if(get_results):
		sim_results = get_sim_results(csv_path = file_path, file_name = file_name_with_version)
		if(delete_file):
			os.remove(file_path)

		return sim_results
	else:
		return []

@ray.remote
def run_attack_sim_ray(attack_duration,attack_magnitude,acc_penetration,attack_penetration,inflow,emission_path,rename_file,get_results=True,delete_file=False):

	flow_params = get_flow_params(attack_duration,attack_magnitude,acc_penetration,inflow,emission_path,attack_penetration)

	exp = Experiment(flow_params)

	exp.run(num_runs=1,convert_to_csv=True)

	if(rename_file):
		files = os.listdir(emission_path)
		for file in files:
			if(file[:3] != 'Dur'):
				csv_path = os.path.join(emission_path,file)

		file_name_no_version = 'Dur_'+str(attack_duration)+'_Mag_'+str(attack_magnitude)+'_Inflow_'+str(inflow)+'_ACCPenetration_'+str(acc_penetration)+'_AttackPenetration_'+str(attack_penetration)

		file_version = 1

		for file in files:
			if(file_name_no_version in file):
				file_version += 1

		file_name_with_version = file_name_no_version+'_ver_'+str(file_version)+'.csv'

		file_path = os.path.join(emission_path,file_name_with_version)


		os.rename(csv_path,file_path)

	if(get_results):
		try:
			sim_results = get_sim_results(csv_path = file_path, file_name = file_name_with_version)
			write_sim_results(sim_results)
		except:
			sim_results = []

		if(delete_file):
			os.remove(file_path)

		return sim_results
	else:
		return []

def iter_run(attack_duration_list,
	attack_magnitude_list,
	acc_penetration_list,
	inflow_list,
	attack_penetration_list,
	emission_path,
	rename_file=True,
	get_results=True,
	delete_file=False,
	want_parallel=True):

	sim_results_list = []
	sim_result_ids = [] #For when parallel with ray

	for inflow in inflow_list:
		for attack_duration in attack_duration_list:
			for attack_magnitude in attack_magnitude_list:
				for acc_penetration in acc_penetration_list:
					for attack_penetration in attack_penetration_list:

						try:
							if(want_parallel):
								#Uses Ray's parallel processing:
								sim_result_ids.append(
									run_attack_sim_ray.remote(
										attack_duration=attack_duration,
										attack_magnitude=attack_magnitude,
										acc_penetration=acc_penetration,
										attack_penetration=attack_penetration,
										inflow=inflow,
										emission_path=emission_path,
										rename_file=rename_file,
										get_results=get_results,
										delete_file=delete_file))
							else:
								sim_results = run_attack_sim(attack_duration,attack_magnitude,acc_penetration,attack_penetration,inflow,emission_path,rename_file)

								sim_results_list.append(sim_results)
						except:
							print('Simulation failed.')

	
	if(want_parallel):
		sim_results_list = ray.get(sim_result_ids)

	print('Simulations finished.')

	return sim_results_list

	# if(want_parallel):
	# 	sim_result_paths = ray.get(sim_result_ids)
	# 	return sim_result_paths
					
					
if __name__ == "__main__":
    
    emission_path = 'i24_adversarial_sims/'
    attack_duration_list = [2,4,6,8,10]
    attack_magnitude_list = [-.2,-.5,-.75,-1.0,-1.25]
 	acc_penetration_list =[.1]
	attack_penetration_list = [.2]
 	inflow_list = [2400]

 	num_runs = 1
	 
	 
	for i in range(num_runs):
		sim_results_list = iter_run(attack_duration_list,
			attack_magnitude_list,
			acc_penetration_list,
			inflow_list,
			attack_penetration_list,
			emission_path,
			rename_file=True,
			get_results=True,
			delete_file=False,
			want_parallel=True)
		
		
	print('Simulations finished.')













	
	















