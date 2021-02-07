
from flow.controllers import FollowerStopper, IDMController, ContinuousRouter, OVMController
from flow.controllers.car_following_adversarial import ACC_Switched_Controller_Attacked
from flow.controllers.lane_change_controllers import StaticLaneChanger
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams
from flow.core.params import SumoCarFollowingParams
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.networks.ring import RingNetwork, ADDITIONAL_NET_PARAMS


# For running a simulation:
from flow.core.experiment import Experiment

# For data processing:
import pandas as pd
import numpy as np
import os
import sys
import time

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
	ring_length,
	emission_path,
	attack_penetration,
	want_render=False,
	display_attack_info=True):

	SIM_LENGTH = 1000

	sim_step = .1 #Simulation step size

	horizon = int(np.floor(SIM_LENGTH/sim_step)) #Number of simulation steps

	WARMUP_STEPS = 1000 #Attack vehicles don't attack before this # of steps

	NUM_VEHICLES = 100

	vehicle_length = 4.0

	##################################
	#ATTACK VEHICLE PARAMETERS:
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

	# Numbers of vehicles:

	num_human_drivers = int(np.ceil(NUM_VEHICLES*(1-acc_penetration)))
	num_acc_drivers = NUM_VEHICLES - num_human_drivers
	num_benign_acc_drivers = int(np.ceil(num_acc_drivers*(1-attack_penetration)))
	num_attack_acc_drivers = num_acc_drivers - num_benign_acc_drivers

	if display_attack_info:
		print('Number of human drivers: '+str(num_human_drivers))
		print('Number of benign acc drivers: '+str(num_benign_acc_drivers))
		print('Number of attack acc drivers: '+str(num_attack_acc_drivers))


	# Initialize network:
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


	flow_params = dict(
		# name of the experiment
		exp_tag='ring',

		# name of the flow environment the experiment is running on
		env_name=AccelEnv,

		# name of the network class the experiment is running on
		network=RingNetwork,

		# simulator that is used by the experiment
		simulator='traci',

		# sumo-related parameters (see flow.core.params.SumoParams)
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
			additional_params=ADDITIONAL_ENV_PARAMS,
		),

		# network-related parameters (see flow.core.params.NetParams and the
		# network's documentation or ADDITIONAL_NET_PARAMS component)
		net=NetParams(
			additional_params={
				"length": ring_length,
				"lanes": 1,
				"speed_limit": 30,
				"resolution": 40,
			}, ),

		# vehicles to be placed in the network at the start of a rollout (see
		# flow.core.params.VehicleParams)
		veh=vehicles,

		# parameters specifying the positioning of vehicles upon initialization/
		# reset (see flow.core.params.InitialConfig)
		initial=InitialConfig(
			shuffle=True,
		),
	)

	return flow_params

def rename_file(csv_path,emission_path,attack_duration,attack_magnitude,acc_penetration,attack_penetration,ring_length):
	files = os.listdir(emission_path)

	file_name_no_version = 'Dur_'+str(attack_duration)+'_Mag_'+str(attack_magnitude)+'_RingLength_'+str(ring_length)+'_ACCPenetration_'+str(acc_penetration)+'_AttackPenetration_'+str(attack_penetration)

	file_version = 1

	for file in files:
		if(file_name_no_version in file):
			file_version += 1

	file_name_with_version = file_name_no_version+'_ver_'+str(file_version)+'.csv'

	file_path = os.path.join(emission_path,file_name_with_version)

	os.rename(csv_path,file_path)

	return file_name_with_version

def run_attack_sim(attack_duration,attack_magnitude,acc_penetration,attack_penetration,ring_length,emission_path,get_results=True,delete_file=True,want_render=False):

	flow_params = get_flow_params(attack_duration,attack_magnitude,acc_penetration,ring_length,emission_path,attack_penetration,want_render=want_render)

	exp = Experiment(flow_params)

	[info_dict,csv_path] = exp.run(num_runs=1,convert_to_csv=True)

	file_name_with_version = rename_file(csv_path,emission_path,attack_duration,attack_magnitude,acc_penetration,attack_penetration,ring_length)
	file_path = os.path.join(emission_path,file_name_with_version)

	sim_results = []

	if(get_results):
		sim_results = get_sim_results(csv_path=file_path,file_name=file_name_with_version)

	if(delete_file):
		os.remove(file_path)

	return sim_results


@ray.remote
def run_attack_sim_ray(attack_duration,attack_magnitude,acc_penetration,attack_penetration,ring_length,emission_path,get_results=True,delete_file=False,want_render=False):

	sim_results = run_attack_sim(attack_duration,attack_magnitude,acc_penetration,attack_penetration,ring_length,emission_path,want_render=want_render,get_results=get_results,delete_file=delete_file)

	return sim_results


def run_attack_batch(attack_duration,
	attack_magnitude,
	acc_penetration,
	attack_penetration,
	ring_length,
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
					ring_length=ring_length,
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
	ring_length_list,
	attack_penetration_list,
	emission_path,
	batch_runs=8,
	get_results=True,
	delete_file=True,
	want_parallel=True,
	csv_name='sim_results_ring_current.csv'):

	sim_results_list = []
	sim_result_ids = [] #For when parallel with ray

	start_time = time.time()

	for ring_length in ring_length_list:
		for attack_duration in attack_duration_list:
			for attack_magnitude in attack_magnitude_list:
				for acc_penetration in acc_penetration_list:
					for attack_penetration in attack_penetration_list:
						sim_result_ids.append(
							run_attack_sim_ray.remote(
								attack_duration=attack_duration,
								attack_magnitude=attack_magnitude,
								acc_penetration=acc_penetration,
								attack_penetration=attack_penetration,
								ring_length=ring_length,
								emission_path=emission_path,
								get_results=True,
								delete_file=True))

	try:
		end_time = time.time()
		compute_time = end_time - start_time

		print('Simulations finished.')
		print('Total computation time: '+str(compute_time))
		print('Time per simulation: '+str(compute_time/len(sim_results_list)))
	except:
		print('simulations Finished.')

	return sim_results_list




if __name__ == "__main__":
	emission_path = 'i24_adversarial_sims/'
	acc_penetration = 0.2
	attack_penetration = 0.2
	attack_magnitude = -.25
	attack_duration = 10.0
	ring_length = (25*100)+400 #s_eq = 15/vehicle + 400 for vehicle lengths

	start_time = time.time()

	sim_results = run_attack_sim(attack_duration,attack_magnitude,acc_penetration,attack_penetration,ring_length,emission_path,get_results=False,delete_file=False,want_render=False)

	end_time = time.time()

	print('Sim results:')
	print(sim_results)

	print('Sim time: '+str(end_time-start_time))

	# ring_length_list = list(np.linspace(25,40,21)*100+400)
	# ring_length_list = [(30*100)+400]
	# acc_penetration_list = [0.2]
	# attack_penetration_list = [0.2]
	# attack_magnitude_list = [-.25,-.5,-.75,-1.0]
	# attack_duration_list = [2.5,5.0,7.5,10.0]

	# sim_results_list = []
	# sim_result_ids = [] #For when parallel with ray

	# start_time = time.time()

	# for ring_length in ring_length_list:
	# 	for attack_duration in attack_duration_list:
	# 		for attack_magnitude in attack_magnitude_list:
	# 			for acc_penetration in acc_penetration_list:
	# 				for attack_penetration in attack_penetration_list:
	# 					sim_result_ids.append(
	# 						run_attack_sim_ray.remote(
	# 							attack_duration=attack_duration,
	# 							attack_magnitude=attack_magnitude,
	# 							acc_penetration=acc_penetration,
	# 							attack_penetration=attack_penetration,
	# 							ring_length=ring_length,
	# 							emission_path=emission_path,
	# 							get_results=True,
	# 							delete_file=True))

	# sim_results_param_sweep = ray.get(sim_result_ids)


	# sim_results_param_sweep = iter_run(attack_duration_list,
	# 	attack_magnitude_list,
	# 	acc_penetration_list,
	# 	ring_length_list,
	# 	attack_penetration_list,
	# 	emission_path,
	# 	batch_runs=1)

	print(sim_results_param_sweep)

	np.savetxt('ring_sim_results_param_sweep.csv',sim_results_param_sweep)







