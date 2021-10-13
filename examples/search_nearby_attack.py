import numpy as np
from load_sim_results import get_sim_data_dict

csv_path = '/Volumes/My Passport for Mac/Dur_10.0_Mag_-1.0_Inflow_2400.0_ACCPenetration_0.2_AttackPenetration_0.2_ver_2.csv'

sim_dict = get_sim_data_dict(csv_path)

veh_ids = list(sim_dict.keys())

mal_ACC_ids = []
for veh_id in veh_ids:
	if(len(sim_dict[veh_id])>0):
		is_mal = False
		for i in range(len(sim_dict[veh_id])):
			if(sim_dict[veh_id][i][-2] == '1'): is_mal=True
		if(is_mal): mal_ACC_ids.append(veh_id)

mal_id = mal_ACC_ids[75]
ACC_data = sim_dict[mal_id]
ACC_data = np.array(ACC_data)
time = ACC_data[:,0].astype('float')
speed = ACC_data[:,4].astype('float')
is_mal = ACC_data[:,-2].astype('float')
rel_pos = ACC_data[:,-6].astype('float')
lane_number = ACC_data[:,-8].astype('float')
edge = ACC_data[:,-9]


pt.scatter(time,rel_pos,c=is_mal)
pt.grid()
pt.show()



attack_time_begin = 1190 #These are exactly correct
attack_time_end = 1200

attack_indices = np.logical_and(time>attack_time_begin,time<attack_time_end)

pt.scatter(time[attack_indices],
	speed[attack_indices],
	c=is_mal[attack_indices])
pt.show()


# Search for other vehicles in time-area zone:
begin_search_time = 1100
end_search_time = 1500
relative_pos_behind = 0
relative_pos_ahead = 600
edge_to_choose = 'Eastbound_3'


selected_data = {}

for veh_id in veh_ids:
	if(len(sim_dict[veh_id]) > 0): #Might be empty
		veh_data = np.array(sim_dict[veh_id])
		#Grad all relevant data entries:

		time = veh_data[:,0].astype('float')
		speed = veh_data[:,4].astype('float')
		is_mal = veh_data[:,-2].astype('float')
		rel_pos = veh_data[:,-6].astype('float')
		lane_number = veh_data[:,-8].astype('float')
		edge = veh_data[:,-9]


		valid_times = np.logical_and(time>begin_search_time,time<end_search_time)
		valid_pos = np.logical_and(rel_pos>relative_pos_behind,rel_pos<relative_pos_ahead)
		valid_edges = edge == edge_to_choose
		valid_points = np.logical_and(valid_times,valid_pos)
		valid_points = np.logical_and(valid_points,valid_edges)

		if(np.sum(valid_points) > 0):
			selected_times = time[valid_points]
			selected_speeds = speed[valid_points]
			selected_pos = rel_pos[valid_points]
			selected_lanes = lane_number[valid_points]

			selected_data[veh_id] = np.array([selected_times,selected_speeds,selected_pos,selected_lanes])

#This could be changed to look at other lanes:
lane_0_data = []
lane_1_data = []
lane_2_data = []
lane_3_data = []
lane_4_data = []

lane_change_actions = []

selected_veh_ids = list(selected_data.keys())

for veh_id in selected_veh_ids:
	veh_data = selected_data[veh_id]

	last_lane = veh_data[3,0]
	
	for i in range(len(veh_data[0,:])):
		lane = veh_data[3,i]

		if(lane == 0):
			lane_0_data.append(np.array(veh_data[:,i]))
		elif(lane == 1):
			lane_1_data.append(np.array(veh_data[:,i]))
		elif(lane == 2):
			lane_2_data.append(np.array(veh_data[:,i]))
		elif(lane == 3):
			lane_3_data.append(np.array(veh_data[:,i]))
		elif(lane == 4):
			lane_4_data.append(np.array(veh_data[:,i]))


		if(lane != last_lane):
			lane_change_actions.append([veh_data[:,i-1],veh_data[:,i]])

		last_lane = lane

#Grab specific mal ACC data:
# mal_id = 'flow_60.26'
# mal_ACC_data = selected_data[mal_id]


selected_mal_ACC_ids = []
for veh_id in selected_veh_ids:
	if(veh_id in mal_ACC_ids):
		selected_mal_ACC_ids.append(veh_id)

lane_1_attack_data = []
lane_2_attack_data = []
lane_3_attack_data = []
lane_4_attack_data = []

for veh_id in selected_mal_ACC_ids:
	veh_data = selected_data[veh_id]
	lane = veh_data[3,0]

	if(lane == 1):
		lane_1_attack_data.append(np.array(veh_data))
	elif(lane == 2):
		lane_2_attack_data.append(np.array(veh_data))
	elif(lane == 3):
		lane_3_attack_data.append(np.array(veh_data))
	elif(lane == 4):
		lane_4_attack_data.append(np.array(veh_data))





# ACC_data = np.array(ACC_data)
# time = ACC_data[:,0].astype('float')
# speed = ACC_data[:,4].astype('float')
# is_mal = ACC_data[:,-2].astype('float')
# rel_pos = ACC_data[:,-6].astype('float')
# lane_number = ACC_data[:,-8].astype('float')
# edge = ACC_data[:,-9]


lane_0_data = np.array(lane_0_data)      
lane_1_data = np.array(lane_1_data)      
lane_2_data = np.array(lane_2_data)
lane_3_data = np.array(lane_3_data)
lane_4_data = np.array(lane_4_data)



dot_size = 3.0
pt.figure(figsize = [10,10])
pt.subplot(1,4,1)
pt.scatter(lane_1_data[:,0],lane_1_data[:,2],c=lane_1_data[:,1],s=dot_size)
for data in lane_1_attack_data:
	pt.plot(data[0,:],data[2,:],'r')
pt.clim([0,30.0])
pt.xlabel('Time [s]')
pt.ylabel('Position')
pt.title('Lane 1')
pt.ylim([relative_pos_behind,relative_pos_ahead])
pt.xlim([begin_search_time,end_search_time])
pt.subplot(1,4,2)
pt.scatter(lane_2_data[:,0],lane_2_data[:,2],c=lane_2_data[:,1],s=dot_size)
for data in lane_2_attack_data:
	pt.plot(data[0,:],data[2,:],'r')
pt.clim([0,30.0])
pt.xlabel('Time [s]')
pt.title('Lane 2')
pt.ylim([relative_pos_behind,relative_pos_ahead])
pt.xlim([begin_search_time,end_search_time])
pt.subplot(1,4,3)
pt.scatter(lane_3_data[:,0],lane_3_data[:,2],c=lane_3_data[:,1],s=dot_size)
for data in lane_3_attack_data:
	pt.plot(data[0,:],data[2,:],'r')
pt.clim([0,30.0])
pt.xlabel('Time [s]')
pt.title('Lane 3')
pt.ylim([relative_pos_behind,relative_pos_ahead])
pt.xlim([begin_search_time,end_search_time])
pt.subplot(1,4,4)
pt.scatter(lane_4_data[:,0],lane_4_data[:,2],c=lane_4_data[:,1],s=dot_size)
for data in lane_4_attack_data:
	pt.plot(data[0,:],data[2,:],'r')
pt.clim([0,30.0])
pt.xlabel('Time [s]')
pt.title('Lane 4')
pt.ylim([relative_pos_behind,relative_pos_ahead])
pt.xlim([begin_search_time,end_search_time])
pt.show()

	
# Plot speeds:

dot_size = 3.0
pt.figure(figsize = [10,10])
pt.subplot(4,1,1)
pt.scatter(lane_1_data[:,0],lane_1_data[:,1],s=dot_size)
pt.ylabel('Speed [m/s]')
pt.title('Lane 1')
pt.ylim([0,30])
pt.subplot(4,1,2)
pt.scatter(lane_2_data[:,0],lane_2_data[:,1],s=dot_size)
pt.title('Lane 2')
pt.ylabel('Speed [m/s]')
pt.ylim([0,30])
pt.subplot(4,1,3)
pt.scatter(lane_3_data[:,0],lane_3_data[:,1],s=dot_size)
pt.title('Lane 3')
pt.ylabel('Speed [m/s]')
pt.ylim([0,30])
pt.subplot(4,1,4)
pt.scatter(lane_4_data[:,0],lane_4_data[:,1],s=dot_size)


pt.plot(mal_ACC_data[0,:],mal_ACC_data[1,:],'r')
pt.ylabel('Speed [m/s]')
pt.xlabel('Time [s]')
pt.title('Lane 4')
pt.ylim([0,30])
pt.show()

















