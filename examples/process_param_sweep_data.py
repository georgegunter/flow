import numpy as np
from load_sim_results import get_sim_data_dict
import sys
sys.path.append('/Users/vanderbilt/Desktop/Research_2020/Traffic_Attack/Anomaly_Detection/ACCIntrusionDetection')
from get_ae_performance import load_ae_model
from utils import sliding_window
import time
import csv

import ray
if(not ray.is_initialized()):
	ray.init(num_cpus=5,ignore_reinit_error=True)

def get_speeds_seperated(csv_path):
	sim_data_dict = get_sim_data_dict(csv_path)
	veh_ids = list(sim_data_dict)

	mal_ACC_ids = []
	for veh_id in veh_ids:
		if(len(sim_dict[veh_id])>0):
			is_mal = False
			for i in range(len(sim_dict[veh_id])):
				if(sim_dict[veh_id][i][-2] == '1'): is_mal=True
			if(is_mal): mal_ACC_ids.append(veh_id)

	benign_ids = []
	for veh_id in veh_ids:
		if(len(sim_dict[veh_id])>0):
			if veh_id not in mal_ACC_ids:
				benign_ids.append(veh_id)

	speeds_benign = []
	for veh_id in benign_ids:
		sys.stdout.write('\r'+veh_id)
		veh_data = np.array(sim_dict[veh_id]) 
		speed_vals = veh_data[:,4].astype('float')
		speeds_benign.append([veh_id,speed_vals])

	speeds_malicious = []
	for veh_id in mal_ACC_ids:
		sys.stdout.write('\r'+veh_id)
		veh_data = np.array(sim_dict[veh_id]) 
		speed_vals = veh_data[:,4].astype('float')
		speeds_malicious.append([veh_id,speed_vals])


	return [speeds_malicious,speeds_benign]



@ray.remote
def get_speeds_seperated_ray(csv_path):
	return get_speeds_seperated(csv_path)

@ray.remote
def get_agg_speed_stats_ray(csv_path):
	[speeds_malicious,speeds_benign] = get_speeds_seperated(csv_path)
	mal_speed_stats = []
	for data in speeds_malicious:
		speed_vals = data[1]
		mal_speed_stats.append([np.mean(speed_vals),np.std(speed_vals)])
	benign_speed_stats =[]
	for data in speeds_benign:
		speed_vals = data[1]
		benign_speed_stats.append([np.mean(speed_vals),np.std(speed_vals)])

	return [mal_speed_stats,benign_speed_stats]




def get_sim_speed_data_ray(csv_path_list):













def check_for_collisions(csv_path):
	row_num = 1

	curr_veh_id = 'id'

	sim_dict = {}

	has_collision = []

	with open(csv_path, newline='') as csvfile:
		
		csvreader = csv.reader(csvfile, delimiter=',')

		for row in csvreader:
			if(row_num>1):
				if(int(row[-1])==1):
					has_collision.append(row[0:1])
			row_num+=1

	return has_collision

def check_all_data_for_collisions(sim_files):
	has_collisions_all = []
	for file in sim_files: 
		sys.stdout.write('\r'+file) 
		has_collision = check_for_collisions(os.path.join(emission_path,file)) 
		if(len(has_collision)>0):
			print()
			print(file+' has collision.') 
			has_collisions_all.append([file,has_collision])
			print() 

	print('Number files with collisions: '+len(has_collisions_all))

	return has_collisions_all



if __name__ == "__main__":
	emission_path = '/Volumes/My Passport for Mac/parameter_sweep'

	sim_files = os.listdir(emission_path)
	sim_files.remove('.DS_Store')    


	print('Finished.')












