import os
import numpy as np
import time

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
if(not ray.is_initialized()):
	ray.init(ignore_reinit_error=True)

def get_flow_params(attack_duration,
	attack_magnitude,
	acc_penetration,
	inflow,
	emission_path,
	attack_penetration,
	want_render=False,
	display_attack_info=True):

	SIM_LENGTH = 2000 #simulation length in seconds

	sim_step = .2 #Simulation step size

	horizon = int(np.floor(SIM_LENGTH/sim_step)) #Number of simulation steps

	WARMUP_STEPS = 1000 #Attack vehicles don't attack before this # of steps

	BASELINE_INFLOW_PER_LANE = inflow #Per lane flow rate in veh/hr

	inflow_speed = 25.5

	ON_RAMP_FLOW = 1000

	highway_start_edge = 'Eastbound_2'

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
		'attack_decel_rate':attack_magnitude,
		'display_attack_info':display_attack_info})

	adversarial_router = (i24_adversarial_router,{})

	#Should never attack, so just a regular ACC:
	benign_ACC_controller = (ACC_Switched_Controller_Attacked,{
		'warmup_steps':horizon,
		'Total_Attack_Duration':0,
		'attack_decel_rate':0,
		'display_attack_info':display_attack_info})

	##################################
	#DRIVER TYPES AND INFLOWS:
	##################################
	# lane_list = ['0','1','2','3']
	lane_list = ['1','2','3','4']

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
				depart_lane=lane,
				depart_speed=inflow_speed)

		if(ACC_BENIGN_INFLOW > 0):
			inflow.add(
				veh_type="benign_ACC",
				edge=highway_start_edge,
				vehs_per_hour=ACC_BENIGN_INFLOW ,
				depart_lane=lane,
				depart_speed=inflow_speed)

	#handles when vehicles wait too long to try and merge and get stuck on merge:
	human_routing_controller = (I24Router,{'position_to_switch_routes':15})

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
			depart_lane=lane,
			depart_speed=inflow_speed)

	inflow.add(
		veh_type="human_on_ramp",
		edge='Eastbound_On_1',
		vehs_per_hour=ON_RAMP_FLOW,
		depart_lane='random',
		depart_speed=20)

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
			render=want_render,
			color_by_speed=False,
			use_ballistic=True,
			emission_path=emission_path,
			print_warnings=False,
			restart_instance=True
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

def rename_file(csv_path,emission_path,attack_duration,attack_magnitude,acc_penetration,attack_penetration,inflow):
	files = os.listdir(emission_path)

	file_name_no_version = 'Dur_'+str(attack_duration)+'_Mag_'+str(attack_magnitude)+'_Inflow_'+str(inflow)+'_ACCPenetration_'+str(acc_penetration)+'_AttackPenetration_'+str(attack_penetration)

	file_version = 1

	for file in files:
		if(file_name_no_version in file):
			file_version += 1

	file_name_with_version = file_name_no_version+'_ver_'+str(file_version)+'.csv'

	file_path = os.path.join(emission_path,file_name_with_version)

	os.rename(csv_path,file_path)

	return file_name_with_version

def run_attack_sim(attack_duration,attack_magnitude,acc_penetration,attack_penetration,inflow,emission_path,get_results=True,delete_file=True,want_render=False):

	flow_params = get_flow_params(attack_duration,attack_magnitude,acc_penetration,inflow,emission_path,attack_penetration,want_render=want_render)

	exp = Experiment(flow_params)

	[info_dict,csv_path] = exp.run(num_runs=1,convert_to_csv=True)

	file_name_with_version = rename_file(csv_path,emission_path,attack_duration,attack_magnitude,acc_penetration,attack_penetration,inflow)
	file_path = os.path.join(emission_path,file_name_with_version)

	sim_results = []

	if(get_results):
		sim_results = get_sim_results(csv_path=file_path,file_name=file_name_with_version)

	if(delete_file):
		os.remove(file_path)

	return sim_results


@ray.remote
def run_attack_sim_ray(attack_duration,attack_magnitude,acc_penetration,attack_penetration,inflow,emission_path,get_results=True,delete_file=False,want_render=False):

	sim_results = run_attack_sim(attack_duration,attack_magnitude,acc_penetration,attack_penetration,inflow,emission_path,want_render=want_render,get_results=get_results,delete_file=delete_file)

	return sim_results


def run_attack_batch(attack_duration,
	attack_magnitude,
	acc_penetration,
	attack_penetration,
	inflow,
	emission_path,
	get_results=True,
	delete_file=False,
	batch_runs=8):

	sim_result_ids = []

	# start_time = time.time()

	for batch in range(batch_runs):
		sim_result_ids.append(
			run_attack_sim_ray.remote(
				attack_duration=attack_duration,
				attack_magnitude=attack_magnitude,
				acc_penetration=acc_penetration,
				attack_penetration=attack_penetration,
				inflow=inflow,
				emission_path=emission_path,
				get_results=get_results,
				delete_file=delete_file))

	# end_time = time.time()
	# batch_sim_time = end_time-start_time
	# print('Batch execution time: '+str(batch_sim_time))
	# print('Time per sim: '+str(batch_sim_time/batch_runs))

	sim_results = ray.get(sim_result_ids)

	return sim_results




def iter_run(attack_duration_list,
	attack_magnitude_list,
	acc_penetration_list,
	inflow_list,
	attack_penetration_list,
	emission_path,
	batch_runs=8,
	get_results=True,
	delete_file=True,
	want_parallel=True,
	csv_name='sim_results_current.csv'):

	sim_results_list = []
	sim_result_ids = [] #For when parallel with ray

	start_time = time.time()

	for inflow in inflow_list:
		for attack_duration in attack_duration_list:
			for attack_magnitude in attack_magnitude_list:
				for acc_penetration in acc_penetration_list:
					for attack_penetration in attack_penetration_list:
						try:
							if(want_parallel):

								sim_results_temp = run_attack_batch(attack_duration,
									attack_magnitude,
									acc_penetration,
									attack_penetration,
									inflow,
									emission_path,
									get_results=get_results,
									delete_file=delete_file,
									batch_runs=batch_runs)

								for sim in sim_results_temp:
									sim_results_list.append(sim)


								# #Uses Ray's parallel processing:
								# sim_result_ids.append(
								# 	run_attack_sim_ray.remote(
								# 		attack_duration=attack_duration,
								# 		attack_magnitude=attack_magnitude,
								# 		acc_penetration=acc_penetration,
								# 		attack_penetration=attack_penetration,
								# 		inflow=inflow,
								# 		emission_path=emission_path,
								# 		get_results=get_results,
								# 		delete_file=delete_file))
							else:
								sim_results = run_attack_sim(attack_duration,attack_magnitude,acc_penetration,attack_penetration,inflow,emission_path,rename_file)

								sim_results_list.append(sim_results)
						except:
							print('Simulation failed.')

						#Save to csv after either every batch, or every sim:
						np.savetxt(csv_name,np.array(sim_results_list))

	try:
		end_time = time.time()
		compute_time = end_time - start_time

		print('Simulations finished.')
		print('Total computation time: '+str(compute_time))
		print('Time per simulation: '+str(compute_time/len(sim_results_list)))
	except:
		print('simulations Finished.')

	return sim_results_list

	# if(want_parallel):
	# 	sim_result_paths = ray.get(sim_result_ids)
	# 	return sim_result_paths
										
if __name__ == "__main__":

	emission_path = 'i24_adversarial_sims/'

	inflow_list = [1200,1800,2400]
	acc_penetration = 0.2
	attack_penetration = 0.2
	attack_magnitude = -.25
	attack_duration = 5.0

	# run_attack_sim(attack_duration,attack_magnitude,acc_penetration,attack_penetration,inflow,emission_path,get_results=False,delete_file=False,want_render=False)


	# sim_result_ids = []

	# start_time = time.time()

	for inflow in inflow_list:
		run_attack_sim(attack_duration,attack_magnitude,acc_penetration,attack_penetration,inflow,emission_path,get_results=False,delete_file=False,want_render=False)

# sim_results = run_attack_batch(attack_duration,
# 	attack_magnitude,
# 	acc_penetration,
# 	attack_penetration,
# 	inflow,
# 	emission_path,
# 	get_results=True,
# 	delete_file=True,
# 	batch_runs=8)

# 	np.savetxt('batch_sim_test.csv',np.array(sim_results))

# attack_duration_list = [2.5,5.0,7.5,10.0]
# attack_magnitude_list = [-.25,-.5,-.75,-1.0]
# acc_penetration_list = [0.3]
# attack_penetration_list = [0.2]
# inflow_list = [1800]

# sim_results_list = iter_run(attack_duration_list,
# 	attack_magnitude_list,
# 	acc_penetration_list,
# 	inflow_list,
# 	attack_penetration_list,
# 	emission_path,
# 	batch_runs=8,
# 	get_results=True,
# 	delete_file=True,
# 	want_parallel=True)
# end_time = time.time()
# 	ray.shutdown()
# 	print('Simulations finished.')













	
	















