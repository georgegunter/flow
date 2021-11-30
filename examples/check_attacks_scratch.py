import numpy as np
import matplotlib.pyplot as pt
import csv
import os
import sys
from load_sim_results import get_sim_params 

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
					if(len(curr_veh_data)>100):
						sim_dict[curr_veh_id] = curr_veh_data
					#Rest where data is being stashed:
					curr_veh_data = []
					curr_veh_id = row[1] # Set new veh id
					#Allocate space for storing:
					#sim_dict[curr_veh_id] = []

				curr_veh_id = row[1]
				time = float(row[0])
				edge = row[-9]
				rel_pos = float(row[-6])

				include_data = (time>300 and edge != 'Eastbound_On_1' and edge != 'Eastbound_Off_2')

				# if(edge == 'Eastbound_8' and rel_pos > 200.0):
				# 	include_data = False #Exclude data near the boundary

				if(include_data):
					curr_veh_data.append(row)
				# sys.stdout.write('\r'+'Veh id: '+curr_veh_id+ ' row: ' +str(row_num)+'\r')
			row_num += 1

		#Add the very last vehicle's information:
		if(len(curr_veh_data)>100):
			sim_dict[curr_veh_id] = curr_veh_data
			# sys.stdout.write('\r'+'Veh id: '+curr_veh_id+ ' row: ' +str(row_num)+'\r')
		print('Data loaded.')
	return sim_dict			

if __name__ == '__main__':
	csv_path = '/Volumes/My Passport for Mac/attack_sweep/Dur_15.0_Mag_-1.5_Inflow_2400_ACCPenetration_0.3_AttackPenetration_0.05_ver_1.csv'
	sim_data_dict = get_sim_data_dict_i24(csv_path)

	veh_ids = list(sim_data_dict.keys())
	mal_veh_ids = []
	mal_val_veh_data = {}
	for veh_id in veh_ids:
		temp_data = np.array(sim_data_dict[veh_id])
		is_mal = temp_data[:,-2].astype(float)
		if(np.sum(is_mal) > 0.0):
			mal_veh_ids.append(veh_id)
			mal_val_veh_data[veh_id]=([temp_data[:,0].astype(float),temp_data[:,4].astype(float),is_mal])


	attack_instances = dict.fromkeys(mal_veh_ids)

	for veh_id in mal_veh_ids:
		data = mal_val_veh_data[veh_id]
		num_samples = len(data[0])

		attack_instances[veh_id] = []

		for i in range(num_samples):
			if(data[2][i]==1.0):
				attack_instances[veh_id].append([data[0][i],data[1][i]])

		attack_instances[veh_id] = np.array(attack_instances[veh_id])

	pt.figure()
	for veh_id in mal_veh_ids:
		pt.plot(attack_instances[veh_id][:,0],attack_instances[veh_id][:,1],'.-')


	# construct space-time diagram to look space-time around an attack:



	#%% Extract results from given time-range:
	begin_search_time = 1100
	end_search_time = 1300
	relative_pos_behind = 300
	relative_pos_ahead = 700
	edge_to_choose = 'Eastbound_3'
	lane_to_choose = 1

	selected_data = {}

	for veh_id in veh_ids:
	    if(len(sim_dict[veh_id]) > 0): #Might be empty

	        veh_data = np.array(sim_dict[veh_id])
	        #Grab all relevant data entries:

	        time = veh_data[:,0].astype('float')
	        speed = veh_data[:,4].astype('float')
	        is_mal = veh_data[:,-2].astype('float')
	        rel_pos = veh_data[:,-6].astype('float')
	        lane_number = veh_data[:,-8].astype('float')
	        edge = veh_data[:,-9]

	        valid_times = np.logical_and(time>begin_search_time,time<end_search_time)
	        valid_pos = np.logical_and(rel_pos>relative_pos_behind,rel_pos<relative_pos_ahead)
	        valid_edges = edge == edge_to_choose
	        valid_lanes = lane_number == lane_to_choose
	        valid_points = np.logical_and(valid_times,valid_pos)
	        valid_points = np.logical_and(valid_points,valid_edges)
	        valid_points = np.logical_and(valid_points,valid_lanes)

	        if(np.sum(valid_points) > 0):
	            selected_times = time[valid_points]
	            selected_speeds = speed[valid_points]
	            selected_pos = rel_pos[valid_points]

	            selected_data[veh_id] = np.array([selected_times,selected_speeds,selected_pos])

	selected_veh_ids = list(selected_data.keys())


	selected_data_list = []
	for veh_id in selected_veh_ids:
	    veh_data = selected_data[veh_id]
	    for i in range(len(veh_data[0,:])):
	        selected_data_list.append(np.array(veh_data[:,i]))

	selected_data_numpy = np.array(selected_data_list)

	times = selected_data_numpy[:,0]
	positions = selected_data_numpy[:,2]
	speeds = selected_data_numpy[:,1]

	dot_size = 7.0
	fontsize = 30
	dpi=150
	pt.figure(figsize = [18,7],dpi=dpi)
	pt.scatter(times,positions,c=speeds,s=dot_size)
	# pt.clim([0,30.0])
	pt.title('GPS penetration rate: 100%',fontsize=fontsize)
	pt.xlabel('Time [s]',fontsize=fontsize)
	pt.ylabel('Position [m]',fontsize=fontsize)
	pt.ylim([relative_pos_behind,relative_pos_ahead])
	pt.xlim([begin_search_time,end_search_time])
	cbar = pt.colorbar(label='Speed [m/s]')
	cbar.ax.tick_params(labelsize=20)
	ax = cbar.ax
	text = ax.yaxis.label
	font = matplotlib.font_manager.FontProperties(size=fontsize)
	text.set_font_properties(font)
	# pt.savefig('spacetime_GPS_dense.png',dpi=dpi) 
	pt.show()













