import numpy as np
import matplotlib.pyplot as pt

import pandas as pd
import sys
import os

import ray 
if(not ray.is_initialized()):
	ray.init(ignore_reinit_error=True)

#%% useful functions for processing flow data:

def get_vehicle_data(csv_path=None,
					 edge_id='Eastbound_7',
					 time_range=[300,800],
					 pos_range=[0,800],
					 print_progress=False):
	'''
	Needs to be given the path to a flow/SUMO emissions file.
	'''
	data_track = pd.read_csv(csv_path,delimiter=',')
	ids = data_track.id.unique() #numpy array
	
	ids = list(ids)
	
	# vehicle_data = dict.fromkeys(ids)
	
	vehicle_data = {}
	
	relevant_fields = \
		['time','speed','headway','leader_id','follower_id','lane_number','relative_position','fuel','is_malicious']
	
	num_ids = len(ids)
	curr_id_num = 1
	
	for id_val in ids:
		if(print_progress):
			sys.stdout.write('\r'+'Loading: '+str(curr_id_num)+'/'+str(num_ids))
		
		data = data_track[(data_track['id'] == id_val) &
				   (data_track['edge_id'] == edge_id) &
				   (data_track['time']>=time_range[0]) &
				   (data_track['time']<=time_range[1]) &
				   (data_track['relative_position']>=pos_range[0]) &
				   (data_track['relative_position']<=pos_range[1])]
		

		if(len(data) > 100):
			vehicle_data[id_val] = dict.fromkeys(relevant_fields)
			
			for field in relevant_fields:
				
				vehicle_data[id_val][field] = np.array(data[field])
			
		curr_id_num += 1

	return vehicle_data

def get_space_time_diagram(data_track,edge_id,lane_number):
	space_time_data = data_track[(data_track['lane_number'] == lane_number) &
				   (data_track['edge_id'] == edge_id)]
	
	time = space_time_data['time']
	position = space_time_data['relative_position']
	speed = space_time_data['speed']*2.237
	   
	pt.figure(figsize=(30,30))
	pt.scatter(time,position,s=5.0,c=speed)
	pt.colorbar()
	pt.title('Space Time, edge: '+str(edge_id)+' lane: '+str(lane_number)+' color: speed [mph]')
	pt.ylabel('Position [m]')
	pt.xlabel('Time [s]')
	pt.clim([0,70])
	
	pt.show()
  
def get_testing_data(vehicle_data,data_type):
	veh_ids = list(vehicle_data.keys())
	testing_data = dict.fromkeys(veh_ids)
	
	if(data_type == 'time-gap'):
		for veh_id in veh_ids:
			speed_vals = vehicle_data[veh_id]['speed']
			headway_vals = vehicle_data[veh_id]['headway']
			
			time_gap_val = np.divide(headway_vals,speed_vals)
			
			testing_data[veh_id] = time_gap_val
			
		
	elif(data_type == 'speed'):
		for veh_id in veh_ids:
			speed_vals = vehicle_data[veh_id]['speed']
			testing_data[veh_id] = speed_vals
			
	return testing_data
	
def get_total_fuel(vehicle_data,want_HVs=True):
	veh_ids = list(vehicle_data.keys())
	vehicle_fuel_consumption = {}
	total_fuel_consumption = 0
	
	for veh_id in veh_ids:
		
		if(want_HVs):
			#Only want to look at the human driven vehicles:
			is_ACC = (len(np.unique(vehicle_data[veh_id]['is_malicious']))>1)
			is_HV = not is_ACC
			if(is_HV):
				fuel_rate = vehicle_data[veh_id]['fuel']
				time = vehicle_data[veh_id]['time']
				
				num_steps = len(time)
				
				fuel_consumption = 0.0
				
				for t in range(1,num_steps):
					dt = time[t]-time[t-1]
					fuel_consumption += dt*fuel_rate[t]
					
				vehicle_fuel_consumption[veh_id] = fuel_consumption
				
				total_fuel_consumption += fuel_consumption   
			
		else:
		
			fuel_rate = vehicle_data[veh_id]['fuel']
			time = vehicle_data[veh_id]['time']
			
			num_steps = len(time)
			
			fuel_consumption = 0.0
			
			for t in range(1,num_steps):
				dt = time[t]-time[t-1]
				fuel_consumption += dt*fuel_rate[t]
				
			vehicle_fuel_consumption[veh_id] = fuel_consumption
			
			total_fuel_consumption += fuel_consumption
		
	return vehicle_fuel_consumption,total_fuel_consumption

def get_total_vehicle_distance(vehicle_data,want_HVs=True):
	veh_ids = list(vehicle_data.keys())
	
	vehicle_distance_traveled = {}
	total_distance_traveled = 0.0
	
	
	
	for veh_id in veh_ids:
		
		if(want_HVs):
			#Only want to look at the human driven vehicles:
			is_ACC = (len(np.unique(vehicle_data[veh_id]['is_malicious']))>1)
			is_HV = not is_ACC
			if(is_HV):
				veh_dist_traveled = vehicle_data[veh_id]['relative_position'][-1] - \
					vehicle_data[veh_id]['relative_position'][0]
				vehicle_distance_traveled[veh_id] = veh_dist_traveled
				total_distance_traveled += veh_dist_traveled
		else:
			veh_dist_traveled = vehicle_data[veh_id]['relative_position'][-1] - \
					vehicle_data[veh_id]['relative_position'][0]
			vehicle_distance_traveled[veh_id] = veh_dist_traveled
			total_distance_traveled += veh_dist_traveled
		
	return vehicle_distance_traveled,total_distance_traveled

def get_total_travel_time(vehicle_data,want_HVs=True):
	veh_ids = list(vehicle_data.keys())
	
	veh_travel_times ={}
	total_travel_time = 0.0
	
	for veh_id in veh_ids:
		
		if(want_HVs):
			#Only want to look at the human driven vehicles:
			is_ACC = (len(np.unique(vehicle_data[veh_id]['is_malicious']))>1)
			is_HV = not is_ACC
			if(is_HV):
				veh_travel_time = vehicle_data[veh_id]['time'][-1] - \
					vehicle_data[veh_id]['time'][0]
				veh_travel_times[veh_id] = veh_travel_time
				total_travel_time += veh_travel_time
		else:
			veh_travel_time = vehicle_data[veh_id]['time'][-1] - \
					vehicle_data[veh_id]['time'][0]
			veh_travel_times[veh_id] = veh_travel_time
			total_travel_time += veh_travel_time
		
	return veh_travel_times,total_travel_time
		
def get_total_speed_variance(vehicle_data,want_HVs=True):
	veh_ids = list(vehicle_data.keys())
	
	veh_speed_vars ={}
	
	all_speeds = []
	
	for veh_id in veh_ids:
		
		if(want_HVs):
			#Only want to look at the human driven vehicles:
			is_ACC = (len(np.unique(vehicle_data[veh_id]['is_malicious']))>1)
			is_HV = not is_ACC
			if(is_HV):
				
				veh_speed = vehicle_data[veh_id]['speed']
				veh_speed_vars[veh_id] = np.var(veh_speed)
				
				for i in range(len(veh_speed)):
					all_speeds.append(veh_speed[i])
					
				
		else:
			veh_speed = vehicle_data[veh_id]['speed']
			veh_speed_vars[veh_id] = np.var(veh_speed)
			
			for i in range(len(veh_speed)):
				all_speeds.append(veh_speed[i])
		
	total_speed_var = np.var(np.array(all_speeds))
	
	return veh_speed_vars,total_speed_var
	
def get_sim_params(file):
	params = []
	num_char = len(file)

	i=0
	while(file[i] != '_'): i+= 1
	i += 1
	j = i
	while(file[j] != '_'): j+= 1
	params.append(float(file[i:j]))
	i = j+1

	while(file[i:i+3] != 'ver'):
		while(file[i] != '_'): i+= 1
		i += 1
		j = i
		while(file[j] != '_'): j+= 1
		params.append(float(file[i:j]))
		i = j+1
	
	return params

@ray.remote
def get_sim_results_ray(csv_path,file_name,print_progress=True):
	print('Loading: '+file_name)
	flow_vehicle_data = get_vehicle_data(csv_path,print_progress=print_progress)

	number_veh = len(flow_vehicle_data)
	number_ACCs = 0
	for veh_id in list(flow_vehicle_data.keys()):
		is_ACC = len(np.unique(flow_vehicle_data[veh_id]['is_malicious'])) > 1
		if is_ACC:
			number_ACCs += 1
		
	_,total_fuel_consumption = get_total_fuel(flow_vehicle_data)
	_,total_distance_traveled = get_total_vehicle_distance(flow_vehicle_data)
	_,total_travel_time = get_total_travel_time(flow_vehicle_data)
	_,total_speed_var = get_total_speed_variance(flow_vehicle_data)
	
	sim_params = get_sim_params(file_name)

	x = []

	for p in sim_params:
		x.append(p)
	x.append(total_fuel_consumption)
	x.append(total_distance_traveled)
	x.append(total_travel_time)
	x.append(total_speed_var)
	x.append(number_veh)
	x.append(number_ACCs)

	return x

def get_sim_results(csv_path,file_name,print_progress=True):
	print('Loading: '+file_name)
	flow_vehicle_data = get_vehicle_data(csv_path,print_progress=print_progress)

	number_veh = len(flow_vehicle_data)
	number_ACCs = 0
	for veh_id in list(flow_vehicle_data.keys()):
		# Note: Because I 'deactivate' ACC attacks by setting their WARMUP_STEPS
		# to be the length of the sim, 'regular' ACCs are not found using this
		# method. This only finds malicious ACCs.
		is_ACC = len(np.unique(flow_vehicle_data[veh_id]['is_malicious'])) > 1
		if is_ACC:
			number_ACCs += 1
		
	_,total_fuel_consumption = get_total_fuel(flow_vehicle_data)
	_,total_distance_traveled = get_total_vehicle_distance(flow_vehicle_data)
	_,total_travel_time = get_total_travel_time(flow_vehicle_data)
	_,total_speed_var = get_total_speed_variance(flow_vehicle_data)
	
	sim_params = get_sim_params(file_name)

	x = []
	for p in sim_params:
		x.append(p)
	x.append(total_fuel_consumption)
	x.append(total_distance_traveled)
	x.append(total_travel_time)
	x.append(total_speed_var)
	x.append(number_veh)
	x.append(number_ACCs)
		
	return x

def write_sim_results(p):
	files = os.listdir()
	if('sim_results.csv' in files):
		sim_results = np.loadtxt('sim_results.csv',delimiter=',')
	else:
		sim_results = []
	sim_results_list = list(sim_results)
	sim_results_list.append(p)
	
	sim_results = np.array(sim_results_list)
	
	np.savetxt('sim_results.csv',sim_results,delimiter=',')
	
	return
	
		

def get_all_sim_results(sims_path,want_parallel=True):
	files = os.listdir(sims_path)

	sim_result_ids = []
	sim_results = []	

	if(want_parallel):
		for file in files:
			if(file[-3:] == 'csv'):  
				
				csv_path = os.path.join(sims_path,file)

				try:
					sim_result = get_sim_results_ray.remote(csv_path,file)
					sim_result_ids.append(sim_result)
				except:
					print('Faulty loading: '+file)

		sim_results_ray = ray.get(sim_result_ids)
		ray.shutdown()
		
		for sim_result in sim_results_ray:
			sim_results.append(sim_result)
		
	else:
		for file in files:
			if(file[-3:] == 'csv'):
				csv_path = os.path.join(sims_path,file)

				try:
					sim_result =  get_sim_results_ray.remote(csv_path,file)
				except:
					sim_result = []
					print('Faulty loading: '+file)

				sim_results.append(sim_result)



	sim_results = np.array(sim_results)
	
	return sim_results
#%% Look at all simulations
# if __name__ == "__main__":
	
# 	sims_path = '/Users/vanderbilt/Desktop/Research_2020/Traffic_Attack/flow/examples/adversarial_sims/'
	
# 	files = os.listdir(sims_path)
# 	csv_files = 
# 	for

# 	sim_result_ids = []
# 	sim_results = []	

# 	if(want_parallel):
# 		for file in files:
# 			if(file[-3:] == 'csv'):  
				
# 				csv_path = os.path.join(sims_path,file)

# 				try:
# 					sim_result = get_sim_results_ray.remote(csv_path,file)
# 					sim_result_ids.append(sim_result)
# 				except:
# 					print('Faulty loading: '+file)
	
	
	
	# sim_files = os.listdir(sims_path)
	
	# sim_result_ids = []
	
	# for file in sim_files:
		
	#	csv_path = os.path.join(sims_path,file)
		
	#	sim_result_ids.append(get_sim_results_ray.remote(csv_path,file))
		
	# sim_results_ray = ray.get(sim_result_ids)
	
	# sim_results = []
	
	# for sim_result in sim_results_ray:
	#	sim_results.append(sim_result)
		
	# sim_results = np.array(sim_results)
		
	
	
		
		
	