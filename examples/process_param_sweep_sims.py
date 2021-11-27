import sys
import os
import numpy as np
from copy import deepcopy
import csv

import ray

def get_sim_data_dict_i24(csv_path):
	row_num = 1
	curr_veh_id = 'id'
	sim_dict = {}
	curr_veh_data = []

	with open(csv_path, newline='') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=',')
		for row in csvreader:
			if(row_num > 1):
				# Don't read header
				if(curr_veh_id != row[1]):
					#Add in new data to the dictionary:
					
					#Store old data:
					if(len(curr_veh_data)>0):
						sim_dict[curr_veh_id] = curr_veh_data
					#Rest where data is being stashed:
					curr_veh_data = []
					curr_veh_id = row[1] # Set new veh id
					#Allocate space for storing:
					#sim_dict[curr_veh_id] = []

				curr_veh_id = row[1]
				time = float(row[0])
				edge = row[-9]

				include_data = (time>300 and edge != 'Eastbound_On_1' and edge != 'Eastbound_Off_2')

				if(include_data):
					curr_veh_data.append(row)
				# sys.stdout.write('\r'+'Veh id: '+curr_veh_id+ ' row: ' +str(row_num)+'\r')
			row_num += 1

		#Add the very last vehicle's information:
		if(len(curr_veh_data)>0):
			sim_dict[curr_veh_id] = curr_veh_data
			# sys.stdout.write('\r'+'Veh id: '+curr_veh_id+ ' row: ' +str(row_num)+'\r')
		print('Data loaded.')
	return sim_dict		

def write_sim_results(csv_path,file_write_path):
	sim_data_dict = get_sim_data_dict_i24(csv_path)

	data_list = []
	veh_ids = list(sim_data_dict.keys())

	for veh_id in veh_ids:
		temp_veh_data = np.array(sim_data_dict[veh_id])
		speeds = temp_veh_data[:,4].astype(float)
		is_malicious_vals = temp_veh_data[:,-2].astype(float)
		mean_speed = np.mean(speeds)
		speed_std = np.std(speeds)
		is_malicious = np.sum(is_malicious_vals)>0

		data_list.append([veh_id,mean_speed,speed_std,is_malicious])

	file = open(file_write_path, 'w+', newline ='')
	with file:
		write = csv.writer(file,delimiter=',')
		write.writerows(data_list)
	print('Written: '+file_write_path)

@ray.remote
def write_sim_results_ray(csv_path,file_write_path):
	write_sim_results(csv_path,file_write_path)


if __name__ == '__main__':

	ray.init(num_cpus=5)

	repo_path = '/Volumes/My Passport for Mac/benign_parameter_sweep/'
	results_repo_path = '/Volumes/My Passport for Mac/benign_parameter_sweep/Sim_Results'
	file_names = os.listdir(repo_path)
	
	csv_paths = []
	csv_file_names = []
	for file in file_names:
		if(file[-3:]=='csv'):
			csv_paths.append(os.path.join(repo_path,file))
			csv_file_names.append(file)

	temp = []

	for i,csv_path in enumerate(csv_paths):
		file_name = csv_file_names[i][:-4]+'_results.csv'
		file_write_path = os.path.join(results_repo_path,file_name)

		temp.append(write_sim_results_ray.remote(csv_path,file_write_path))

	temp_data = ray.get(temp)

	print('Finished writing all files.')