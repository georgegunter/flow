import numpy as np
import matplotlib.pyplot as pt

import pandas as pd
import sys
import os
import csv

#%%
# import ray 
# if(not ray.is_initialized()):
# 	ray.init(ignore_reinit_error=True)

#%% useful functions for processing flow data:

def get_vehicle_data(csv_path=None,
					 data_frame=None,
					 edge_id='Eastbound_7',
					 time_range=[300,2000],
					 pos_range=[0,800],
					 print_progress=False):
	'''
	Needs to be given the path to a flow/SUMO emissions file.
	'''
	if(data_frame is None):
		data_frame = pd.read_csv(csv_path,delimiter=',')
		
	ids = data_frame.id.unique() #numpy array
	
	ids = list(ids)
	
	# vehicle_data = dict.fromkeys(ids)
	
	vehicle_data = {}
	
	# NOTE: I took out the collision check because it wasn't included in some 
	# should be added back into relevant_fields again...
	
	relevant_fields = \
		['time','speed','headway','leader_id','follower_id','lane_number','edge_id','relative_position','distance','fuel','is_malicious','is_collision']
	
	num_ids = len(ids)
	curr_id_num = 1
	
	for id_val in ids:
		if(print_progress):
			sys.stdout.write('\r'+'Loading: '+str(curr_id_num)+'/'+str(num_ids))
		
		# data = data_frame[(data_frame['id'] == id_val) &
		# 		   (data_frame['edge_id'] == edge_id) &
		# 		   (data_frame['time']>=time_range[0]) &
		# 		   (data_frame['time']<=time_range[1]) &
		# 		   (data_frame['relative_position']>=pos_range[0]) &
		# 		   (data_frame['relative_position']<=pos_range[1])]
		

		data = data_frame[(data_frame['id'] == id_val) &
				   (data_frame['edge_id'] != 'Eastbound_On_1') &
				   (data_frame['edge_id'] != 'Eastbound_Off_2') &
				   (data_frame['time']>=time_range[0]) &
				   (data_frame['time']<=time_range[1])]

		if(len(data) > 100):
			vehicle_data[id_val] = dict.fromkeys(relevant_fields)
			
			for field in relevant_fields:
				
				vehicle_data[id_val][field] = np.array(data[field])
				
			vehicle_data[id_val]['is_acc'] = len(np.unique(vehicle_data[id_val]['is_malicious'])) > 1
# 			vehicle_data[id_val]['has_collision'] = len(np.unique(vehicle_data[id_val]['is_collision'])) > 1
			
		curr_id_num += 1

	return vehicle_data

#%% In work:

# def get_vehicle_data_ray(csv_path=None,
# 					 data_frame=None,
# 					 edge_id='Eastbound_7',
# 					 time_range=[300,2000],
# 					 pos_range=[0,800],
# 					 print_progress=False):
    
    

# @ray.remote
# def ray_select_vehicle_data(data_frame,id_val):
# 		
# 		# data = data_frame[(data_frame['id'] == id_val) &
# 		# 		   (data_frame['edge_id'] == edge_id) &
# 		# 		   (data_frame['time']>=time_range[0]) &
# 		# 		   (data_frame['time']<=time_range[1]) &
# 		# 		   (data_frame['relative_position']>=pos_range[0]) &
# 		# 		   (data_frame['relative_position']<=pos_range[1])]
# 		

# 	data = data_frame[(data_frame['id'] == id_val) &
# 				   (data_frame['edge_id'] != 'Eastbound_On_1') &
# 				   (data_frame['edge_id'] != 'Eastbound_Off_2') &
# 				   (data_frame['time']>=time_range[0]) &
# 				   (data_frame['time']<=time_range[1])]

#     vehicle_data_curr = {}


# 		if(len(data) > 100):
# 			vehicle_data[id_val] = dict.fromkeys(relevant_fields)
# 			
# 			for field in relevant_fields:
# 				
# 				vehicle_data[id_val][field] = np.array(data[field])
# 				
# 			vehicle_data[id_val]['is_acc'] = len(np.unique(vehicle_data[id_val]['is_malicious'])) > 1
# 			vehicle_data[id_val]['has_collision'] = len(np.unique(vehicle_data[id_val]['is_collision'])) > 1
#%%
def get_space_time_diagram(data_frame,edge_id,lane_number):
	space_time_data = data_frame[(data_frame['lane_number'] == lane_number) &
				   (data_frame['edge_id'] == edge_id)]
	
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


def get_mainline_inflow(vehicle_data,inflow_edge):
    veh_ids = list(vehicle_data.keys())
    num_veh = 0
    for veh_id in veh_ids:
        if(vehicle_data[veh_id]['edge_id'][0]==inflow_edge):num_veh += 1
    return num_veh
    


def find_time_integral(times,dx_dt):
	x = 0
	num_samples = len(dx_dt)

	for i in range(num_samples-1):
		dt = times[i+1]-times[i]
		x += dx_dt[i]*dt

	return x
	
def get_total_fuel(vehicle_data,want_HVs=False):
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
	
def get_per_vehicle_speed_data(vehicle_data):
	return None

def get_total_vehicle_distance(vehicle_data,want_HVs=False):

	#Should rewrite to work for the ring...


	veh_ids = list(vehicle_data.keys())
	
	vehicle_distance_traveled = {}
	total_distance_traveled = 0.0
	
	for veh_id in veh_ids:
		
		if(want_HVs):
			#Only want to look at the human driven vehicles:
			is_ACC = (len(np.unique(vehicle_data[veh_id]['is_malicious']))>1)
			is_HV = not is_ACC
			if(is_HV):
				veh_dist_traveled = vehicle_data[veh_id]['distance'][-1] - vehicle_data[veh_id]['relative_position'][0]
				vehicle_distance_traveled[veh_id] = veh_dist_traveled
				total_distance_traveled += veh_dist_traveled
		else:
			veh_dist_traveled = vehicle_data[veh_id]['distance'][-1] - vehicle_data[veh_id]['relative_position'][0]
			vehicle_distance_traveled[veh_id] = veh_dist_traveled
			total_distance_traveled += veh_dist_traveled
		
	return vehicle_distance_traveled,total_distance_traveled

def get_total_travel_time(vehicle_data,want_HVs=False):
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
		
def get_total_speed_variance(vehicle_data,want_HVs=False):
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


def get_sim_results(csv_path,file_name,print_progress):
	print('Loading: '+file_name)
	vehicle_data = get_vehicle_data(csv_path,print_progress=print_progress)
	
	number_veh = len(vehicle_data)
	number_ACCs = 0


	mal_ACCs = []

	for veh_id in list(vehicle_data.keys()):
		is_ACC = len(np.unique(vehicle_data[veh_id]['is_malicious'])) > 1
		if is_ACC:
			mal_ACCs.append(veh_id)
			number_ACCs += 1

	
	# fuel_effs = []
	mean_speeds = []
	var_speeds = []
	veh_ids = list(vehicle_data.keys())
	
	# has_collision = False
	
	for veh_id in veh_ids:

		if(veh_id not in mal_ACCs):

			speed = vehicle_data[veh_id]['speed']
			mean_speed = np.mean(speed)
			speed_var = np.var(speed)
			mean_speeds.append(mean_speed)
			var_speeds.append(speed_var)
		
		# total_fuel = find_time_integral(vehicle_data[veh_id]['time'],vehicle_data[veh_id]['fuel'])
		# total_distance = vehicle_data[veh_id]['distance'][-1]-vehicle_data[veh_id]['distance'][0]
		# fuel_eff = total_distance/total_fuel/1000
		# fuel_effs.append(fuel_eff)
		
		# if vehicle_data[veh_id]['has_collision']:
		#	 has_collision = True
		
	
	# sim_params = get_sim_params(file_name)

	# x = []
	# for p in sim_params:
	#	 x.append(p)
	# x.append(np.mean(mean_speeds))
	# x.append(np.mean(var_speeds))
	# x.append(np.mean(fuel_effs))
	# x.append(number_veh)
	# x.append(number_ACCs)
	# x.append(has_collision)
	
	return [np.mean(mean_speeds),np.mean(var_speeds)]
		

# @ray.remote
# def get_sim_results_ray_helper(csv_path,file_name,print_progress):
# 	return get_sim_results(csv_path,file_name,print_progress)


def get_sim_results_list(csv_path_list,file_name_list,print_progress=True):

	sim_result_ids = []
	num_sims = len(csv_path_list)

	for i in range(num_sims):
		file_name = file_name_list[i]
		csv_path = csv_path_list[i]

		sim_result_ids.append(get_sim_results_ray_helper.remote(csv_path,file_name,print_progress))


	sim_results = ray.get(sim_result_ids)

	return sim_results


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

def get_speed_stats(csv_path):
	#NOTE: add in checks for vehicles to see if they are compromised ACCs or if 
	#a collision occurred. Currently just grabs everything and anything.

	row_num = 1
	veh_id_nums = 0

	curr_veh_id = 'id'
	speeds = []

	results = []

	with open(csv_path, newline='') as csvfile:
		
		csvreader = csv.reader(csvfile, delimiter=',')

		for row in csvreader:

			if(row_num > 1):
				# Don't read header

				if(curr_veh_id != row[1] and row_num > 2):
					#Switched to a new veh_id:
					veh_id_nums += 1
					if(len(speeds) > 0):
						results.append([curr_veh_id,np.mean(speeds),np.sqrt(np.var(speeds))])
						sys.stdout.write('\r'+'Veh id: '+str(veh_id_nums))

					speeds = [] # Reset speeds
					curr_veh_id = row[1] # Set new veh id
					is_mal_ACC = False
					has_collision = False

				# Add data from new row:
				# sys.stdout.write('\r'+'Veh id: '+str(veh_id_nums) + ' row: ' +str(row_num))

				curr_veh_id = row[1]
				time = float(row[0])
				edge = row[13]

				include_data = (time>300 and edge != 'Eastbound_On_1' and edge != 'Eastbound_Off_2')

				if(include_data):

					# sys.stdout.write('\r'+'Veh id: '+str(veh_id_nums) + ' row: ' +str(row_num))
					speed = float(row[4])
					speeds.append(speed)

			row_num += 1 #Keep track of which row is being looked at

	return results		

def get_sim_results_csv(csv_path):

	results = get_speed_stats(csv_path)

	mean_speeds = [] 
	std_speeds = [] 
	for x in results: 
		mean_speeds.append(x[1])
		std_speeds.append(x[2])

	return [len(mean_speeds),np.mean(mean_speeds),np.mean(std_speeds)]


def write_to_master_csv(sim_results,file_name):
	params = get_sim_params(file_name)

	x = []
	for p in params: x.append(p) 
	for s in sim_results: x.append(s) 
	
	all_results = np.loadtxt('sim_results_master.csv')

	all_results_list = list(all_results)

	all_results_list.append(x)

	all_results = np.array(all_results_list)

	np.savetxt('sim_results_master.csv',all_results)


def write_results_to_csv(sim_results,file_name,csv_repo_path='/Users/vanderbilt/Desktop/Research_2020/Traffic_Attack/flow/examples/i24_adversarial_sims/results_csv_repo'):
	
	params = get_sim_params(file_name)

	x = []
	for p in params: x.append(p) 
	for s in sim_results: x.append(s)
	x = np.array(x)

	files = os.listdir(csv_repo_path)

	file_name_no_version = 'Dur_'+str(params[0])+'_Mag_'+str(params[1])+'_Inflow_'+str(params[2])+'_ACCPenetration_'+str(params[3])+'_AttackPenetration_'+str(params[4])

	i = -5
	while(file_name[i] != '_'): i -= 1

	file_version = 1

	for file in files:
		if(file_name_no_version in file):
			file_version += 1

	# file_version +=1

	file_name_with_version = file_name_no_version+'_ver_'+str(file_version)+'.csv'

	file_path = os.path.join(csv_repo_path,file_name_with_version)

	np.savetxt(file_path,x)

def get_sim_data_dict(csv_path):
	# Returns a dictionary with lists of data sorted by vehicle:

	row_num = 1

	curr_veh_id = 'id'

	sim_dict = {}

	curr_veh_data = []

	with open(csv_path, newline='') as csvfile:
		
		csvreader = csv.reader(csvfile, delimiter=',')

		for row in csvreader:

			if(row_num > 1):
				# Don't read header

				if(curr_veh_id != row[1] and row_num > 2):
					#Switched to a new veh_id:

					#Add in new data to the dictionary:
					# sys.stdout.write('\r'+'Veh id: '+str(veh_id_nums) + ' row: ' +str(row_num))
					
					#Store old data:
					if(len(curr_veh_data)>0):
						sim_dict[curr_veh_id] = curr_veh_data

					#Rest where data is being stashed:
					curr_veh_data = []

					curr_veh_id = row[1] # Set new veh id
					#Allocate space for storing:
					sim_dict[curr_veh_id] = []

				curr_veh_id = row[1]
				time = float(row[0])
				edge = row[13]

				include_data = (time>300 and edge != 'Eastbound_On_1' and edge != 'Eastbound_Off_2')

				if(include_data):
					#data we do want to keep track of:
					curr_veh_data.append(row)

			row_num += 1

	return sim_dict



def write_results_platoon_to_csv(sim_results,file_name,csv_repo_path='/Users/vanderbilt/Desktop/Research_2020/Traffic_Attack/flow/examples/i24_adversarial_sims/results_platoon_csv_repo'):
	
	# Currently just a copy of the above code...

	params = get_sim_params(file_name)

	x = []
	for p in params: x.append(p) 
	for s in sim_results: x.append(s)
	x = np.array(x)

	files = os.listdir(csv_repo_path)

	file_name_no_version = 'Dur_'+str(attack_duration)+'_Mag_'+str(attack_magnitude)+'_Inflow_'+str(inflow)+'_ACCPenetration_'+str(acc_penetration)+'_AttackPenetration_'+str(attack_penetration)

	i = -5
	while(file_name[i] != '_'): i -= 1

	file_version = int(file_name[i+1:-4])

	for file in files:
		if(file_name_no_version in file):
			file_version += 1

	file_name_with_version = file_name_no_version+'_ver_'+str(file_version)+'.csv'

	file_path = os.path.join(csv_repo_path,file_name_with_version)

	np.savetxt(file_path,x)

def get_all_params(emission_path):
	files = os.listdir(emission_path)
	unique_params = []


	for file in files:
		if(file[-3:]=='csv'):
			params = get_sim_params(file)

			if(params not in unique_params):
				unique_params.append(params)

	return unique_params






