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

	csv_repo_path = '/Volumes/My Passport for Mac/parameter_sweep/Sim_Results'
	csv_files = os.listdir(csv_repo_path)

	sim_data_dict = dict.fromkeys(csv_files)

	for file in csv_files: 
		csv_path = os.path.join(csv_repo_path,file) 
		agg_res = get_agg_sim_res(csv_path) 
		params = get_sim_params(file) 
		sim_data_dict[file] = [params,agg_res] 

	low_cong = [[],[],[]]
	med_cong = [[],[],[]]
	high_cong = [[],[],[]]

	for file in csv_files: 
		params = sim_data_dict[file][0] 
		agg_res = sim_data_dict[file][1]

		if(params[2] == 1200.0):
			low_cong[0].append(agg_res[0])
			low_cong[1].append(agg_res[1])
			low_cong[2].append(agg_res[2]) 
		if(params[2] == 1800.0):
			med_cong[0].append(agg_res[0])
			med_cong[1].append(agg_res[1])
			med_cong[2].append(agg_res[2]) 
		if(params[2] == 2400.0):
			high_cong[0].append(agg_res[0])
			high_cong[1].append(agg_res[1])
			high_cong[2].append(agg_res[2]) 

	fontsize=20
	dotsize = 15
	pt.figure() 
	pt.subplot(3,1,1)
	pt.plot(low_cong[0],'g.',markersize=dotsize,label = 'Low congestion')
	pt.plot(med_cong[0],'b.',markersize=dotsize,label = 'Medium congestion')
	pt.plot(high_cong[0],'r.',markersize=dotsize,label = 'High congestion')
	pt.legend(fontsize=fontsize)
	pt.ylabel('Mean speed [m/s]',fontsize=fontsize)
	pt.subplot(3,1,2)
	pt.plot(low_cong[1],'g.',markersize=dotsize)
	pt.plot(med_cong[1],'b.',markersize=dotsize)
	pt.plot(high_cong[1],'r.',markersize=dotsize)
	pt.ylabel('Mean std. speed [m/s]',fontsize=fontsize)

	pt.subplot(3,1,3)
	pt.plot(low_cong[2],'g.',markersize=dotsize)
	pt.plot(med_cong[2],'b.',markersize=dotsize)
	pt.plot(high_cong[2],'r.',markersize=dotsize)
	pt.ylabel('Throughput [number veh]',fontsize=fontsize)



