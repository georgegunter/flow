from flow.controllers import FollowerStopper, IDMController, ContinuousRouter, OVMController
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams
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

def get_flow_dict(v_des,model_params,emission_path):

	alpha = model_params[0]
	beta = model_params[1]
	v_max = model_params[2]
	s_star = model_params[3]
	s0 = model_params[4]

	human_accel = (OVMController,{'alpha':alpha,'beta':beta,'v_max':v_max,'s_star':s_star,'s0':s0,'noise':.1})

	vehicles = VehicleParams()
	vehicles.add(
	    veh_id="human",
	    acceleration_controller=human_accel,
	    routing_controller=(ContinuousRouter, {}),
	    num_vehicles=20)

	vehicles.add(
	    color='red',
	    veh_id="AV",
	    acceleration_controller=(FollowerStopper, {'v_des':v_des}),
	    routing_controller=(ContinuousRouter, {}),
	    num_vehicles=1)


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
	        render=False,
	        sim_step=0.1,
	        emission_path=emission_path,
	    ),

	    # environment related parameters (see flow.core.params.EnvParams)
	    env=EnvParams(
	        horizon=3000,
	        warmup_steps=750,
	        additional_params=ADDITIONAL_ENV_PARAMS,
	    ),

	    # network-related parameters (see flow.core.params.NetParams and the
	    # network's documentation or ADDITIONAL_NET_PARAMS component)
	    net=NetParams(
	        additional_params={
	            "length": 260,
	            "lanes": 1,
	            "speed_limit": 30,
	            "resolution": 40,
	        }, ),

	    # vehicles to be placed in the network at the start of a rollout (see
	    # flow.core.params.VehicleParams)
	    veh=vehicles,

	    # parameters specifying the positioning of vehicles upon initialization/
	    # reset (see flow.core.params.InitialConfig)
	    # initial=InitialConfig(
	    #     bunching=20,
	    # ),
	)

	return flow_params

def run_sim(v_des,model_params,emission_path):

	flow_params = get_flow_dict(v_des,model_params,emission_path)

	exp = Experiment(flow_params)

	[emission_files,info_dict] = exp.run(num_runs=1,convert_to_csv=True)

	csv_path = emission_files[0]

	return csv_path

def get_sim_results(csv_path):
	data = pd.read_csv(csv_path,delimiter=',')
	ids = data.id.unique() #numpy array
	ids = list(ids)

	sim_time = np.array(data[data['id']==ids[0]]['time'])
	sim_length = sim_time[-1]

	time_threshold = sim_length/2

	speed_measurements = data[data['time'] > time_threshold]['speed']

	speed_measurements = np.array(speed_measurements)

	ave_speed = np.mean(speed_measurements)

	std_speed = np.std(speed_measurements)

	return [ave_speed,std_speed]


if __name__ == "__main__":
	emission_path = '/Users/vanderbilt/Desktop/Research_2020/CIRCLES/Official_Flow/flow/examples/follower_stopper_sims/'

	model_params = [0.6660,21.5975,8.9368,2.2146,2.8150]

	sim_results = []

	v_des_vals = np.linspace(1.0,9.0,25)
	v_des_vals = list(v_des_vals)

	start_time = time.time()

	for v_des in v_des_vals:
		sys.stdout.write('\r'+'Simulating v_des: '+str(v_des))

		csv_path = run_sim(v_des,model_params,emission_path)

		sim_data = get_sim_results(csv_path)

		sim_results.append([v_des,sim_data[0],sim_data[1]])

		os.remove(csv_path)

	sim_time = time.time() - start_time

	sim_results = np.array(sim_results)

	np.savetxt('follower_stopper_sweep.csv',sim_results)

	print('Simulation sweep finished, time to complete: '+str(sim_time))











