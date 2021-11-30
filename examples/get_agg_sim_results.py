import numpy as np
import matplotlib.pyplot as pt
import csv
import os
import sys



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