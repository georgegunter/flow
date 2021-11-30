import numpy as np
import matplotlib.pyplot as pt
import csv
import os
import sys
from load_sim_results import get_sim_params 

# script for looking at data coming from simulations

def get_agg_sim_res(csv_path):

	mean_speeds = []
	std_speeds = []
	num_veh = 0

	with open(csv_path, newline='') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=',')
		for row in csvreader:
			if(row[-1]=='False'):
				mean_speeds.append(float(row[1]))
				std_speeds.append(float(row[2]))
			num_veh += 1

	mean_speed = np.mean(mean_speeds)
	mean_std_speed = np.mean(std_speeds)
	throughput = num_veh
	return [mean_speed,mean_std_speed,throughput]


if __name__ == '__main__':

	csv_repo_path = '/Volumes/My Passport for Mac/attack_sweep/Sim_Results'
	csv_files = os.listdir(csv_repo_path)

	sim_data_dict = dict.fromkeys(csv_files)

	a_attack_vals = []
	t_attack_vals = []


	for file in csv_files: 
		csv_path = os.path.join(csv_repo_path,file) 
		agg_res = get_agg_sim_res(csv_path) 
		params = get_sim_params(file) 
		sim_data_dict[file] = [params,agg_res]

		if(params[0] not in t_attack_vals): t_attack_vals.append(params[0])
		if(params[1] not in a_attack_vals): a_attack_vals.append(params[1])

	a_attack_vals = np.array(a_attack_vals)
	a_attack_vals.sort()
	t_attack_vals = np.array(t_attack_vals)
	t_attack_vals.sort()

	A,T = np.meshgrid(a_attack_vals,t_attack_vals)

	MTS = np.zeros_like(A)
	MTS_STD = np.zeros_like(A)
	THROUGHPUT = np.zeros_like(A)

	for file in csv_files:
		agg_res = sim_data_dict[file][1]
		params = sim_data_dict[file][0]
		t_attack = params[0]
		a_attack = params[1]

		i = 0
		while(a_attack != a_attack_vals[i]):i+=1

		j=0
		while(t_attack != t_attack_vals[j]):j+=1

		MTS[j,i] = agg_res[0]
		MTS_STD[j,i] = agg_res[1]
		THROUGHPUT[j,i] = agg_res[2]











