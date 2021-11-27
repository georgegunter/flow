import numpy as np
from load_sim_results import get_sim_data_dict
import sys
sys.path.append('/Users/vanderbilt/Desktop/Research_2020/Traffic_Attack/Anomaly_Detection/ACCIntrusionDetection')

from get_ae_performance import load_ae_model

from utils import sliding_window,sliding_window_mult_feat
from cnn_lstm_ae import CNNRecurrentAutoencoder
from train_cnn_lstm_ae import train_CNN_LSTM_AE 
import time
from copy import deepcopy

import csv

def get_measured_leader(ring_sim_dict,veh_id_curr,all_vehicle_ids_measured):
	curr_leader = ring_sim_dict[veh_id_curr][0][6]
	while(curr_leader not in(all_vehicle_ids_measured)):
		curr_leader = ring_sim_dict[curr_leader][0][6]
	return curr_leader

def get_rel_dist_to_measured_leader(ring_sim_dict,veh_id_curr,measured_leader):

	curr_leader = ring_sim_dict[veh_id_curr][0][6]
	temp_data = np.array(ring_sim_dict[veh_id_curr])
	total_spacing = temp_data[:,5].astype('float')
	while(curr_leader != measured_leader):
		temp_data = np.array(ring_sim_dict[curr_leader])
		next_spacing = temp_data[:,5].astype('float')
		total_spacing += next_spacing
		curr_leader = ring_sim_dict[curr_leader][0][6]
	return total_spacing

def get_vel_of_measured_leader(ring_sim_dict,veh_id_curr,measured_leader):

	curr_leader = ring_sim_dict[veh_id_curr][0][6]
	temp_data = np.array(ring_sim_dict[veh_id_curr])
	while(curr_leader != measured_leader):
		temp_data = np.array(ring_sim_dict[curr_leader])
		curr_leader = ring_sim_dict[curr_leader][0][6]
	temp_data = np.array(ring_sim_dict[curr_leader])
	effective_leader_speed = temp_data[:,4].astype('float')
	return effective_leader_speed

def get_sim_data_dict_ring(csv_path):
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
					sim_dict[curr_veh_id] = []

				curr_veh_id = row[1]
				time = float(row[0])
				if(time > 50):
					curr_veh_data.append(row)
				sys.stdout.write('\r'+'Veh id: '+curr_veh_id+ ' row: ' +str(row_num)+'\r')
			row_num += 1

		#Add the very last vehicle's information:
		sim_dict[curr_veh_id] = curr_veh_data
		sys.stdout.write('\r'+'Veh id: '+curr_veh_id+ ' row: ' +str(row_num)+'\r')
		print('Data loaded.')
	return sim_dict	

def get_all_losses_4feat(model,sim_data_dict,all_vehicle_ids_measured):
	veh_ids = veh_ids = list(sim_data_dict.keys())
	losses_list = []

	for veh_id in veh_ids:
		measured_leader = get_measured_leader(benign_sim_dict,veh_id,all_vehicle_ids_measured)
		effective_spacing = get_rel_dist_to_measured_leader(benign_sim_dict,veh_id,measured_leader)
		effective_leader_speed = get_vel_of_measured_leader(benign_sim_dict,veh_id,measured_leader)
		speed_sample = temp_veh_data[:,4].astype(float)
		accel_sample = np.gradient(speed_sample,.1)
		effective_rel_speed_sample = effective_leader_speed- speed_sample

		timeseries_list = []
		timeseries_list.append(speed_sample)
		timeseries_list.append(accel_sample)
		timeseries_list.append(effective_spacing)
		timeseries_list.append(effective_rel_speed_sample)

		sys.stdout.write('\r'+'Veh id: '+veh_id)
		[reconstructions,losses] = sliding_window_mult_feat(model,timeseries_list)

		losses_list.append(losses)

	return losses_list

def plot_time_space(sim_data_dict):
	return None

def get_loss_dict(model,sim_data_dict):
	veh_ids = list(sim_data_dict.keys())
	losses_dict = dict.fromkeys(veh_ids)

	num_veh_ids = len(veh_ids)
	veh_ids_processed = 0

	for veh_id in veh_ids:

		timeseries_list = []

		temp_veh_data = np.array(sim_data_dict[veh_id])
		measured_leader = get_measured_leader(sim_data_dict,veh_id,veh_ids)
		effective_spacing = get_rel_dist_to_measured_leader(sim_data_dict,veh_id,measured_leader)
		effective_leader_speed = get_vel_of_measured_leader(sim_data_dict,veh_id,measured_leader)

		speed_sample = temp_veh_data[:,4].astype(float)
		accel_sample = np.gradient(speed_sample,.1) #The time step value:
		effective_spacing_sample = effective_spacing
		effective_rel_speed_sample = effective_leader_speed - speed_sample

		timeseries_list.append(speed_sample)
		timeseries_list.append(accel_sample)
		timeseries_list.append(effective_spacing_sample)
		timeseries_list.append(effective_rel_speed_sample)

		[reconstructions,losses] = sliding_window_mult_feat(model,timeseries_list)
		testing_losses_dict[veh_id] = losses
		veh_ids_processed = veh_ids_processed + 1

		sys.stdout.write('\r'+'Veh id: '+veh_id + ' '+str(veh_ids_processed)+'/'+str(num_veh_ids))

	return losses_dict

def get_loss_filter(losses_dict,sim_data_dict,loss_window_length=100):
	#Something in here throws an error...
	veh_ids = list(losses_dict.keys())

	losses_filtered_dict = dict.fromkeys(veh_ids)

	for veh_id in veh_ids:
		losses = losses_dict[veh_id]
		temp_veh_data = np.array(sim_data_dict[veh_id])
		veh_losses_filtered = np.zeros_like(temp_veh_data[:,0])
		loss_counts = np.zeros_like(temp_veh_data[:,0])

		for i in range(len(losses)):
			l = losses[i]
			veh_losses_filtered[i:loss_window_length] = veh_losses_filtered[i:loss_window_length] + l
			loss_counts[i:loss_window_length] = loss_counts[i:loss_window_length] + 1

		return losses_filtered_dict

def get_ring_positions(sim_data_dict,ring_length):
	veh_ids = list(sim_data_dict.keys())
	ring_positions = dict.fromkeys(veh_ids)

	edge_length = ring_length/4.0

	for veh_id in veh_ids:
		temp_veh_data = np.array(sim_data_dict[veh_id])

		init_edge = temp_veh_data[0,-9]
		init_rel_position = temp_veh_data[0,-6].astype(float)

		distances = temp_veh_data[:,-7].astype(float)

		distances = distances - distances[0]

		init_dist = 0

		# Find initial distance along all edges:
		if(init_edge=='right'):
			init_dist = init_rel_position
		elif(init_edge=='top'):
			init_dist = init_rel_position + edge_length
		elif(init_edge=='left'):
			init_dist = init_rel_position + 2*edge_length
		elif(init_edge=='bottom'):
			init_dist = init_rel_position + 3*edge_length

		distances = distances + init_dist

		ring_positions[veh_id] = distances

	return ring_positions

def plot_losses_timespace(losses_dict,sim_data_dict,ring_length,loss_window_length=100):

	losses_filtered_dict = get_loss_filter(losses_dict,sim_data_dict,loss_window_length)
	ring_positions = get_ring_positions(sim_data_dict,ring_length)

	veh_ids_measured = list(losses_filtered_dict.keys())

	times_list = [] 
	pos_list = [] 
	loss_list = [] 
 
	for veh_id in veh_ids_measured:
		#Look only at losses for those vehicles being measured:

		temp_veh_data = np.array(sim_data_dict[veh_id]) 
		time = temp_veh_data[:,0].astype(float) 
		loss_filt = losses_filtered_dict[veh_id] 
		ring_pos = ring_positions[veh_id]
		for i in range(len(time)): 
			times_list.append(time[i]) 
			loss_list.append(loss_filt[i]) 
			pos_list.append(ring_pos[i]) 
	
	print('Stacked data')

	pt.figure()
	pt.scatter(times_list,np.mod(pos_list,ring_length),c=loss_list,s=2.0)
	pt.ylabel('Positions [m]',fontsize=30)
	pt.xlabel('Time [s]',fontsize=30)
	pt.colorbar(label='Loss')
	pt.title('Strong attack, 100 GPS penetraiton',fontsize=30)
	pt.show()
	pt.ylim([0,ring_length])
	pt.xlim([float(temp_veh_data[0,0]),float(temp_veh_data[-1,0])])

def save_losses(sim_data_dict,losses_dict,loss_file_name):
	loss_list = []
	veh_ids = list(sim_data_dict.keys())
	for veh_id in veh_ids:
		loss_list.append([])
		losses = losses_dict[veh_id]
		for i in range(len(losses)):
			loss_list.append(losses[i])

	with open(loss_file_name, 'w', newline='') as csvfile:
		csv_writer = csv.writer(csvfile, delimiter=',')
		for row in loss_list:
			csv_writer.writerow(row)
		print('Saved '+loss_file_name)

def get_GPS_measurements_ring(sim_data_dict_ring,GPS_penetration_rate):

	veh_ids = list(sim_data_dict_ring.keys())
	num_measured_vehicle_ids = int(np.floor(len(veh_ids)*GPS_penetration_rate))
	measured_veh_ids = deepcopy(veh_ids)

	#Want to make sure malicious vehicles are included in measurements:
	mal_ACCs = []
	for veh_id in veh_ids:
		if("acc_adversarial" in veh_id):
			mal_ACCs.append(veh_id)
			num_measured_vehicle_ids -= 1
			measured_veh_ids.remove(veh_id)

	for i in range(len(measured_veh_ids)-num_measured_vehicle_ids):
		rand_int = np.random.randint(0,len(measured_veh_ids))
		del measured_veh_ids[rand_int]

	all_vehicle_ids_measured = []

	for veh_id in mal_ACCs:
		all_vehicle_ids_measured.append(veh_id)
	for veh_id in measured_veh_ids:
		all_vehicle_ids_measured.append(veh_id)

	GPS_data_dict = dict.fromkeys(all_vehicle_ids_measured)

	for veh_id in all_vehicle_ids_measured:

		temp_veh_data = np.array(sim_data_dict_ring[veh_id])
		measured_leader = get_measured_leader(sim_data_dict_ring,veh_id,all_vehicle_ids_measured)
		effective_spacing = get_rel_dist_to_measured_leader(sim_data_dict_ring,veh_id,measured_leader)
		effective_leader_speed = get_vel_of_measured_leader(sim_data_dict_ring,veh_id,measured_leader)



		speed_sample = temp_veh_data[:,4].astype(float)
		accel_sample = np.gradient(speed_sample,.1) #The time step value:
		effective_spacing_sample = effective_spacing
		effective_rel_speed_sample = effective_leader_speed - speed_sample

		GPS_data_dict[veh_id] = [speed_sample,accel_sample,effective_spacing_sample,effective_rel_speed_sample]

	return GPS_data_dict

def get_test_losses(GPS_penetration_rate,
	test_sim_dict,
	MODEL_SAVE_PATH)

	test_veh_ids  = list(test_sim_dict.keys())

	GPS_data_dict = get_GPS_measurements_ring(test_sim_dict,GPS_penetration_rate)

	measured_veh_ids = list(GPS_data_dict.keys())

	#Model hyper-parameters:
	n_features = 4
	embedding_dim = 32
	cnn_channels = 8
	kernel_size = 16
	stride = 1
	batch_size = 16
	device = 'cpu'
	seq_len = 100

	model = CNNRecurrentAutoencoder(seq_len, n_features, embedding_dim, cnn_channels, kernel_size, stride, device)
	model = model.load_state_dict(torch.load(MODEL_SAVE_PATH))

	test_losses_dict = dict.fromkeys(measured_veh_ids)

	veh_ids_processed = 0
	num_veh_ids = len(measured_veh_ids)

	for veh_id in measured_veh_ids:

		timeseries_list = GPS_data_dict[veh_id] 
		[reconstructions,losses] = sliding_window_mult_feat(model,timeseries_list)
		test_losses_dict[veh_id] = losses
		sys.stdout.write('\r'+'Veh id: '+veh_id + ' '+str(veh_ids_processed)+'/'+str(num_veh_ids))

	return test_losses_dict

def train_detector(GPS_penetration_rate,train_sim_dict,MODEL_SAVE_PATH):
	return None


if __name__ == '__main__':
	benign_file_path = '/Users/vanderbilt/Desktop/Research_2020/Traffic_Attack/flow/examples/ringroad_adversarial_sims/Dur_0.0_Mag_0.0_RingLength_1800_ACCPenetration_0.2_AttackPenetration_0.001_ver_1.csv'
	benign_sim_dict_ring = get_sim_data_dict_ring(benign_file_path)

	benign_veh_ids = list(benign_sim_dict_ring.keys())


	acc_veh_ids = []
	human_veh_ids = []
	for veh_id in veh_ids:
		if(veh_id[:3]=='acc'): acc_veh_ids.append(veh_id)
		else: human_veh_ids.append(veh_id)

	want_50_percent_GPS = False
	if(want_50_percent_GPS):
		# Get a cert		GPS_penetration_rate = 0.50
		num_measured_vehicle_ids = int(np.floor(len(human_veh_ids)*GPS_penetration_rate))
		measured_veh_ids = deepcopy(human_veh_ids)
		for i in range(len(measured_veh_ids)-num_measured_vehicle_ids):
			rand_int = np.random.randint(0,len(measured_veh_ids))
			del measured_veh_ids[rand_int]
		
		all_vehicle_ids_measured = []
		for veh_id in acc_veh_ids:
			all_vehicle_ids_measured.append(veh_id)
		for veh_id in measured_veh_ids:
			all_vehicle_ids_measured.append(veh_id)ain penetration rate of GPS tracked vehicles:


		# Extract data to perform training on:
		num_samples_per_veh = 100
		training_data_list = []

		seq_len = 100

		print('Extracting measurements...')

		for veh_id in all_vehicle_ids_measured:
			sample_start_points = np.random.randint(6000,size=num_samples_per_veh)
			# Begin by just considering speed, acceleration, and measured spacing:
			temp_veh_data = np.array(benign_sim_dict[veh_id])
			measured_leader = get_measured_leader(benign_sim_dict,veh_id,all_vehicle_ids_measured)
			effective_spacing = get_rel_dist_to_measured_leader(benign_sim_dict,veh_id,measured_leader)
			effective_leader_speed = get_vel_of_measured_leader(benign_sim_dict,veh_id,measured_leader)
			for i in range(num_samples_per_veh):

				sys.stdout.write('\r'+'Veh id: '+veh_id+' sample number: '+str(i))

				start = sample_start_points[i]
				end = start+seq_len

				speed_sample = temp_veh_data[start:end,4].astype(float)
				accel_sample = np.gradient(speed_sample,.1) #The time step value:
				effective_spacing_sample = effective_spacing[start:end]
				effective_rel_speed_sample = effective_leader_speed[start:end] - speed_sample

				data_sample = np.zeros([4*seq_len,1])
				data_sample[0:seq_len] = speed_sample.reshape(seq_len,1)
				data_sample[seq_len:2*seq_len] = accel_sample.reshape(seq_len,1)
				data_sample[2*seq_len:3*seq_len] = effective_spacing_sample.reshape(seq_len,1)
				data_sample[3*seq_len:] = effective_rel_speed_sample.reshape(seq_len,1)

				training_data_list.append(data_sample)
				 
		print()
		print('Training data found.')

		train_X = torch.Tensor(np.array(training_data_list))

		model_file_name = 'cnn_lstm_ae_ring_1800_GPS_50_4feat'

		# ring_cnn_lstm_ae = train_CNN_LSTM_AE(train_X,model_file_name)

		n_features=4
		n_epoch=150
		save_path='/Users/vanderbilt/Desktop/Research_2020/Traffic_Attack/Anomaly_Detection/ACCIntrusionDetection/models/'

		embedding_dim = 32
		cnn_channels = 8
		kernel_size = 16
		stride = 1
		batch_size = 16
		device = 'cpu'

		seq_len = 100 #Not good practice to hardcode, but this probably won't need to change...

		print(f"Model name: {model_file_name}")
		trainset = SeqDataset(train_X)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

		model = CNNRecurrentAutoencoder(seq_len, n_features, embedding_dim, cnn_channels, kernel_size, stride, device)

		optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

		np.savez(f'{save_path}cnn_lstm_ae_{model_file_name}.npz', seq_len=seq_len, embedding_dim=embedding_dim,
				 cnn_channels=cnn_channels, kernel_size=kernel_size, stride=stride)

		SAVE_PATH = f'{save_path}cnn_lstm_ae_{model_file_name}.pt'
		
		do_load = False
		if do_load and os.path.exists(SAVE_PATH):
			model.load_state_dict(torch.load(SAVE_PATH))

		if not do_load:
			best_loss = 999999
		else:
			train_ls, train_tot = eval_data(model=model, dataloader=trainloader)
			best_loss = 1.0 * train_ls/train_tot

		for e in range(n_epoch):
			l = train_epoch(model=model, optimizer=optimizer, dataloader=trainloader)
			train_ls, train_tot = eval_data(model=model, dataloader=trainloader)
			avg_loss = 1.0 * train_ls / train_tot
			if e % 10 == 0:
				print("Epoch %d, total loss %f, total predictions %d, avg loss %f" % (e, train_ls, train_tot, avg_loss),
					  datetime.datetime.now())
			if avg_loss < best_loss:
				best_loss = avg_loss
				torch.save(model.state_dict(), SAVE_PATH)
				print("Saving model. Best loss: "+str(best_loss))

	want_75_percent_GPS = False
	if(want_75_percent_GPS):			
		#Redo, except now look at 75% GPS penetration rate:
		print('Training detector at 75 percent GPS penetration rate.')

		# Get a certain penetration rate of GPS tracked vehicles:
		GPS_penetration_rate = 0.75
		num_measured_vehicle_ids = int(np.floor(len(human_veh_ids)*GPS_penetration_rate))
		measured_veh_ids = deepcopy(human_veh_ids)
		for i in range(len(measured_veh_ids)-num_measured_vehicle_ids):
			rand_int = np.random.randint(0,len(measured_veh_ids))
			del measured_veh_ids[rand_int]
		
		all_vehicle_ids_measured = []
		for veh_id in acc_veh_ids:
			all_vehicle_ids_measured.append(veh_id)
		for veh_id in measured_veh_ids:
			all_vehicle_ids_measured.append(veh_id)

		# Extract data to perform training on:
		num_samples_per_veh = 100
		training_data_list = []

		seq_len = 100

		print('Extracting measurements...')

		for veh_id in all_vehicle_ids_measured:
			sample_start_points = np.random.randint(6000,size=num_samples_per_veh)
			# Begin by just considering speed, acceleration, and measured spacing:
			temp_veh_data = np.array(benign_sim_dict[veh_id])
			measured_leader = get_measured_leader(benign_sim_dict,veh_id,all_vehicle_ids_measured)
			effective_spacing = get_rel_dist_to_measured_leader(benign_sim_dict,veh_id,measured_leader)
			effective_leader_speed = get_vel_of_measured_leader(benign_sim_dict,veh_id,measured_leader)
			for i in range(num_samples_per_veh):

				sys.stdout.write('\r'+'Veh id: '+veh_id+' sample number: '+str(i))

				start = sample_start_points[i]
				end = start+seq_len

				speed_sample = temp_veh_data[start:end,4].astype(float)
				accel_sample = np.gradient(speed_sample,.1) #The time step value:
				effective_spacing_sample = effective_spacing[start:end]
				effective_rel_speed_sample = effective_leader_speed[start:end] - speed_sample

				data_sample = np.zeros([4*seq_len,1])
				data_sample[0:seq_len] = speed_sample.reshape(seq_len,1)
				data_sample[seq_len:2*seq_len] = accel_sample.reshape(seq_len,1)
				data_sample[2*seq_len:3*seq_len] = effective_spacing_sample.reshape(seq_len,1)
				data_sample[3*seq_len:] = effective_rel_speed_sample.reshape(seq_len,1)

				training_data_list.append(data_sample)
				 
		print()
		print('Training data found.')

		train_X = torch.Tensor(np.array(training_data_list))

		model_file_name = 'cnn_lstm_ae_ring_1800_GPS_75_4feat'

		n_features=4
		n_epoch=150
		save_path='/Users/vanderbilt/Desktop/Research_2020/Traffic_Attack/Anomaly_Detection/ACCIntrusionDetection/models/'

		embedding_dim = 32
		cnn_channels = 8
		kernel_size = 16
		stride = 1
		batch_size = 16
		device = 'cpu'

		seq_len = 100 #Not good practice to hardcode, but this probably won't need to change...

		print(f"Model name: {model_file_name}")
		trainset = SeqDataset(train_X)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

		model = CNNRecurrentAutoencoder(seq_len, n_features, embedding_dim, cnn_channels, kernel_size, stride, device)

		optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

		np.savez(f'{save_path}cnn_lstm_ae_{model_file_name}.npz', seq_len=seq_len, embedding_dim=embedding_dim,
				 cnn_channels=cnn_channels, kernel_size=kernel_size, stride=stride)

		SAVE_PATH = f'{save_path}cnn_lstm_ae_{model_file_name}.pt'
		
		do_load = False
		if do_load and os.path.exists(SAVE_PATH):
			model.load_state_dict(torch.load(SAVE_PATH))

		if not do_load:
			best_loss = 999999
		else:
			train_ls, train_tot = eval_data(model=model, dataloader=trainloader)
			best_loss = 1.0 * train_ls/train_tot

		for e in range(n_epoch):
			l = train_epoch(model=model, optimizer=optimizer, dataloader=trainloader)
			train_ls, train_tot = eval_data(model=model, dataloader=trainloader)
			avg_loss = 1.0 * train_ls / train_tot
			if e % 10 == 0:
				print("Epoch %d, total loss %f, total predictions %d, avg loss %f" % (e, train_ls, train_tot, avg_loss),
					  datetime.datetime.now())
			if avg_loss < best_loss:
				best_loss = avg_loss
				print("Saving model. Best loss: "+str(best_loss))
				torch.save(model.state_dict(), SAVE_PATH)

	want_100_percent_GPS = True
	if(want_100_percent_GPS):
		#Redo, except now look at 100% GPS penetration rate:
		print('Training detector at 100 percent GPS penetration rate.')

		# Measure all vehicles:
		all_vehicle_ids_measured = veh_ids

		# Extract data to perform training on:
		num_samples_per_veh = 100
		training_data_list = []

		seq_len = 100

		print('Extracting measurements...')

		for veh_id in all_vehicle_ids_measured:
			sample_start_points = np.random.randint(6000,size=num_samples_per_veh)
			# Begin by just considering speed, acceleration, and measured spacing:
			temp_veh_data = np.array(benign_sim_dict[veh_id])
			measured_leader = get_measured_leader(benign_sim_dict,veh_id,all_vehicle_ids_measured)
			effective_spacing = get_rel_dist_to_measured_leader(benign_sim_dict,veh_id,measured_leader)
			effective_leader_speed = get_vel_of_measured_leader(benign_sim_dict,veh_id,measured_leader)

			for i in range(num_samples_per_veh):

				sys.stdout.write('\r'+'Veh id: '+veh_id+' sample number: '+str(i))

				start = sample_start_points[i]
				end = start+seq_len

				speed_sample = temp_veh_data[start:end,4].astype(float)
				accel_sample = np.gradient(speed_sample,.1) #The time step value:
				effective_spacing_sample = effective_spacing[start:end]
				effective_rel_speed_sample = effective_leader_speed[start:end] - speed_sample

				data_sample = np.zeros([4*seq_len,1])
				data_sample[0:seq_len] = speed_sample.reshape(seq_len,1)
				data_sample[seq_len:2*seq_len] = accel_sample.reshape(seq_len,1)
				data_sample[2*seq_len:3*seq_len] = effective_spacing_sample.reshape(seq_len,1)
				data_sample[3*seq_len:] = effective_rel_speed_sample.reshape(seq_len,1)

				training_data_list.append(data_sample)
				 
		print()
		print('Training data found.')

		train_X = torch.Tensor(np.array(training_data_list))

		model_file_name = 'cnn_lstm_ae_ring_1800_GPS_100_4feat'

		n_features=4
		n_epoch=150
		save_path='/Users/vanderbilt/Desktop/Research_2020/Traffic_Attack/Anomaly_Detection/ACCIntrusionDetection/models/'

		embedding_dim = 32
		cnn_channels = 8
		kernel_size = 16
		stride = 1
		batch_size = 16
		device = 'cpu'

		seq_len = 100 #Not good practice to hardcode, but this probably won't need to change...

		print(f"Model name: {model_file_name}")
		trainset = SeqDataset(train_X)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

		model = CNNRecurrentAutoencoder(seq_len, n_features, embedding_dim, cnn_channels, kernel_size, stride, device)

		optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

		np.savez(f'{save_path}cnn_lstm_ae_{model_file_name}.npz', seq_len=seq_len, embedding_dim=embedding_dim,
				 cnn_channels=cnn_channels, kernel_size=kernel_size, stride=stride)

		SAVE_PATH = f'{save_path}cnn_lstm_ae_{model_file_name}.pt'
		
		do_load = False
		if do_load and os.path.exists(SAVE_PATH):
			model.load_state_dict(torch.load(SAVE_PATH))

		if not do_load:
			best_loss = 999999
		else:
			train_ls, train_tot = eval_data(model=model, dataloader=trainloader)
			best_loss = 1.0 * train_ls/train_tot

		for e in range(n_epoch):
			l = train_epoch(model=model, optimizer=optimizer, dataloader=trainloader)
			train_ls, train_tot = eval_data(model=model, dataloader=trainloader)
			avg_loss = 1.0 * train_ls / train_tot
			if e % 10 == 0:
				print("Epoch %d, total loss %f, total predictions %d, avg loss %f" % (e, train_ls, train_tot, avg_loss),
					  datetime.datetime.now())
			if avg_loss < best_loss:
				best_loss = avg_loss
				print("Saving model. Best loss: "+str(best_loss))
				torch.save(model.state_dict(), SAVE_PATH)


	#%% Look at training results:

	GPS_penetration_rate = '50'


	SAVE_PATH = '/Users/vanderbilt/Desktop/Research_2020/Traffic_Attack/Anomaly_Detection/ACCIntrusionDetection/models/cnn_lstm_ae_cnn_lstm_ae_ring_1800_GPS_'+GPS_penetration_rate+'_4feat.pt'
	
	#Model hyper_parameters:
	n_features=4
	embedding_dim = 32
	cnn_channels = 8
	kernel_size = 16
	stride = 1
	batch_size = 16
	device = 'cpu'
	seq_len = 100 

	model = CNNRecurrentAutoencoder(seq_len, n_features, embedding_dim, cnn_channels, kernel_size, stride, device)
	model.load_state_dict(torch.load(SAVE_PATH))

	num_veh_ids = len(veh_ids)
	veh_ids_processed = 0

	training_losses_dict = dict.fromkeys(veh_ids)
	for veh_id in veh_ids:

		timeseries_list = []

		temp_veh_data = np.array(benign_sim_dict[veh_id])
		measured_leader = get_measured_leader(benign_sim_dict,veh_id,veh_ids)
		effective_spacing = get_rel_dist_to_measured_leader(benign_sim_dict,veh_id,measured_leader)
		effective_leader_speed = get_vel_of_measured_leader(benign_sim_dict,veh_id,measured_leader)

		speed_sample = temp_veh_data[:,4].astype(float)
		accel_sample = np.gradient(speed_sample,.1) #The time step value:
		effective_spacing_sample = effective_spacing
		effective_rel_speed_sample = effective_leader_speed - speed_sample

		timeseries_list.append(speed_sample)
		timeseries_list.append(accel_sample)
		timeseries_list.append(effective_spacing_sample)
		timeseries_list.append(effective_rel_speed_sample)

		[reconstructions,losses] = sliding_window_mult_feat(model,timeseries_list)
		training_losses_dict[veh_id] = losses
		veh_ids_processed = veh_ids_processed + 1
		sys.stdout.write('\r'+'Veh id: '+veh_id + ' '+str(veh_ids_processed)+'/'+str(num_veh_ids))

	pt.figure()
	for veh_id in veh_ids:
		losses = training_losses_dict[veh_id]
		if('human' in veh_id):
			pt.subplot(2,1,1)
			pt.plot(losses)
		else:
			pt.subplot(2,1,2)
			pt.plot(losses)

	#%% Look at testing results:
	attack_file_path = '/Users/vanderbilt/Desktop/Research_2020/Traffic_Attack/flow/examples/ringroad_adversarial_sims/Dur_10.0_Mag_-1.0_RingLength_1800_ACCPenetration_0.2_AttackPenetration_0.05_ver_1.csv'
	attack_sim_dict = get_sim_data_dict_ring(attack_file_path)
	attack_veh_ids  = list(attack_sim_dict.keys())

	begin_time = time.time()

	num_veh_ids = len(attack_veh_ids)
	veh_ids_processed = 0


	testing_losses_dict = dict.fromkeys(attack_veh_ids)# where to store losses

	for veh_id in attack_veh_ids:

		timeseries_list = []

		temp_veh_data = np.array(attack_sim_dict[veh_id])
		measured_leader = get_measured_leader(attack_sim_dict,veh_id,attack_veh_ids)
		effective_spacing = get_rel_dist_to_measured_leader(attack_sim_dict,veh_id,measured_leader)
		effective_leader_speed = get_vel_of_measured_leader(attack_sim_dict,veh_id,measured_leader)

		speed_sample = temp_veh_data[:,4].astype(float)
		accel_sample = np.gradient(speed_sample,.1) #The time step value:
		effective_spacing_sample = effective_spacing
		effective_rel_speed_sample = effective_leader_speed - speed_sample

		timeseries_list.append(speed_sample)
		timeseries_list.append(accel_sample)
		timeseries_list.append(effective_spacing_sample)
		timeseries_list.append(effective_rel_speed_sample)

		[reconstructions,losses] = sliding_window_mult_feat(model,timeseries_list)
		testing_losses_dict[veh_id] = losses
		veh_ids_processed = veh_ids_processed + 1

		sys.stdout.write('\r'+'Veh id: '+veh_id + ' '+str(veh_ids_processed)+'/'+str(num_veh_ids))

	end_time = time.time()

	print('total computation time: '+str(end_time-begin_time))


	#%% Plot results:

	want_comp_subplots = True

	if(want_comp_subplots):
		pt.figure()

		for veh_id in benign_veh_ids:
			losses = training_losses_dict[veh_id]
			if('human' in veh_id):
				pt.subplot(2,2,1)
				pt.plot(losses)
			else:
				pt.subplot(2,2,2)
				pt.plot(losses)
		pt.subplot(2,2,1)
		pt.ylabel('Loss')
		pt.title('Training humans')
		pt.subplot(2,2,2)
		pt.title('Training ACCs')

		for veh_id in attack_veh_ids:
			losses = testing_losses_dict[veh_id]
			if('human' in veh_id):
				pt.subplot(2,2,3)
				pt.plot(losses)
			else:
				pt.subplot(2,2,4)
				pt.plot(losses)

		pt.subplot(2,2,3)
		pt.xlabel('Time-step')
		pt.ylabel('Loss')
		pt.title('Testing humans')
		pt.subplot(2,2,4)
		pt.xlabel('Time-step')
		pt.title('Testing ACCs')

	#%% plot on the ring-road:

	want_ring_road_plot = True

	if(want_ring_road_plot)

		losses_filtered_dict = 

		times_list = [] 
		pos_list = [] 
		loss_list = [] 
	 
		for veh_id in attack_veh_ids: 
			temp_veh_data = np.array(sim_data_dict[veh_id]) 
			time = temp_veh_data[:,0].astype(float) 
			loss_filt = losses_filtered_dict[veh_id] 
			ring_pos = ring_positions[veh_id]
			for i in range(len(time)): 
				times_list.append(time[i]) 
				loss_list.append(loss_filt[i]) 
				pos_list.append(ring_pos[i]) 
		
		print('Stacked data')

		pt.figure()
		pt.scatter(times_list,np.mod(pos_list,ring_length),c=loss_list,s=2.0)
		pt.ylabel('Positions [m]',fontsize=30)
		pt.xlabel('Time [s]',fontsize=30)
		pt.colorbar(label='Loss')
		pt.title('Strong attack, 100 GPS penetraiton',fontsize=30)
		pt.show()
		pt.ylim([0,ring_length])
		pt.xlim([float(temp_veh_data[0,0]),float(temp_veh_data[-1,0])])




