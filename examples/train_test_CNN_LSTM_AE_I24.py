import numpy as np
import matplotlib.pyplot as pt
from load_sim_results import get_sim_data_dict
import sys
sys.path.append('/Users/vanderbilt/Desktop/Research_2020/Traffic_Attack/Anomaly_Detection/ACCIntrusionDetection')

import torch

from get_ae_performance import load_ae_model

from utils import sliding_window,sliding_window_mult_feat,eval_data,train_epoch,SeqDataset
from cnn_lstm_ae import CNNRecurrentAutoencoder
from train_cnn_lstm_ae import train_CNN_LSTM_AE 
import time
from copy import deepcopy

import csv
import datetime

import ray

def get_effective_leader_measurements(sim_dict,veh_id_curr,all_vehicle_ids_measured):

	#PROBLEM: Has trouble with boundary conditions where vehicles are leaving on the off-ramps

	temp_data = np.array(sim_dict[veh_id_curr])
	
	effective_leader_speed_np_arr = np.zeros_like(temp_data[:,0].astype(float))
	effective_spacing_np_arr = np.zeros_like(temp_data[:,0].astype(float))

	for i in range(len(temp_data[:,0])):

		time = float(temp_data[i,0]) #Need to correlate across same time for different vehicles

		effective_spacing = float(temp_data[i,5]) #Whatever the current spacing is.

		curr_leader = sim_dict[veh_id_curr][i][6]

		j=0 #Index to correlate times

		num_missed_leaders = 0

		while(curr_leader not in(all_vehicle_ids_measured) and curr_leader != ''):
			#Hop to the next leader:

			#find right time:
			try:
				temp_leader_data = np.array(sim_dict[curr_leader])
				leader_times = temp_leader_data[:,0].astype(float)
				j = 0 #index for leader which corrsesponds to same time
				while(leader_times[j] != time):
					j+=1

				if(curr_leader != ''):
					#Add in the accrued spacing from the missed vehicle:
					effective_spacing += float(sim_dict[curr_leader][j][5])
				else:
					#If no leader found then this is the default:
					effective_spacing = 1000.0


				curr_leader = sim_dict[curr_leader][j][6]
				num_missed_leaders +=1
			except:
				curr_leader = '' 

		if(curr_leader == ''):
			#there was no effective leader:
			effective_leader_speed_np_arr[i] = float(temp_data[i][4]) #No speed difference
			effective_spacing_np_arr[i] = 1000.0 # default for no leader

		else:
			try:
				#Need to recorelate times:
				temp_leader_data = np.array(sim_dict[curr_leader])
				leader_times = temp_leader_data[:,0].astype(float)
				j = 0 #index for leader which corrsesponds to same time
				while(leader_times[j] != time):
					j+=1
				#effective leader exists
				effective_leader_speed_np_arr[i] = float(sim_dict[curr_leader][j][4])
				effective_spacing_np_arr[i] = float(effective_spacing)
			except:
				#This may be triggered by the leader entering an exit ramp:
				effective_leader_speed_np_arr[i] = float(temp_data[i][4]) #No speed difference
				effective_spacing_np_arr[i] = 1000.0 # default for no leader

		effective_spacing_np_arr = effective_spacing_np_arr.astype(float)
		effective_leader_speed_np_arr = effective_leader_speed_np_arr.astype(float)


	return effective_leader_speed_np_arr,effective_spacing_np_arr

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

def get_loss_dict(model,sim_data_dict):
	veh_ids = list(sim_data_dict.keys())
	losses_dict = dict.fromkeys(veh_ids)

	num_veh_ids = len(veh_ids)
	veh_ids_processed = 0

	for veh_id in veh_ids:

		timeseries_list = []

		temp_veh_data = np.array(sim_data_dict[veh_id])

		effective_leader_speed,effective_spacing = get_effective_leader_measurements(sim_dict,veh_id,all_vehicle_ids_measured)

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
	veh_ids = list(sim_data_dict.keys())

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

def save_losses(sim_data_dict,losses_dict,loss_file_name):
	# This doesn't really work...
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

def get_GPS_measurements_I24(sim_data_dict,GPS_penetration_rate,want_malicious_veh=True):

	veh_ids = list(sim_data_dict.keys())
	num_measured_vehicle_ids = int(np.floor(len(veh_ids)*GPS_penetration_rate))
	measured_veh_ids = deepcopy(veh_ids)

	#Want to make sure malicious vehicles are included in measurements:
	mal_ACCs = []
	if(want_malicious_veh):
		for veh_id in veh_ids:
			temp_veh_data = np.array(sim_data_dict[veh_id])
			is_malicious = np.sum(temp_veh_data[:,-2].astype(float)) > 0.0
			if(is_malicious):
				measured_veh_ids.append(veh_id)

	for i in range(len(measured_veh_ids)-num_measured_vehicle_ids):
		rand_int = np.random.randint(0,len(measured_veh_ids))
		del measured_veh_ids[rand_int]

	all_vehicle_ids_measured = []

	if(want_malicious_veh):
		for veh_id in mal_ACCs:
			all_vehicle_ids_measured.append(veh_id)
	for veh_id in measured_veh_ids:
		all_vehicle_ids_measured.append(veh_id)

	GPS_data_dict = dict.fromkeys(all_vehicle_ids_measured)

	total_num_vehicle_ids = len(all_vehicle_ids_measured)
	veh_ids_measured = 0

	for veh_id in all_vehicle_ids_measured:

		temp_veh_data = np.array(sim_data_dict[veh_id])

		effective_spacing,effective_leader_speed = get_effective_leader_measurements(sim_data_dict,veh_id,all_vehicle_ids_measured)

		speed_sample = temp_veh_data[:,4].astype(float)
		accel_sample = np.gradient(speed_sample,.2) #The time step value:
		effective_spacing_sample = effective_spacing
		effective_rel_speed_sample = effective_leader_speed - speed_sample

		GPS_data_dict[veh_id] = [speed_sample,accel_sample,effective_spacing_sample,effective_rel_speed_sample]

		veh_ids_measured += 1

		sys.stdout.write('\r'+str(veh_ids_measured)+'/'+str(total_num_vehicle_ids))


	return GPS_data_dict


if __name__ == '__main__':
	benign_file_path = '/Volumes/My Passport for Mac/benign_parameter_sweep/Dur_0.0_Mag_0.0_Inflow_2400_ACCPenetration_0.2_AttackPenetration_0.001_ver_1.csv'
	benign_sim_dict = get_sim_data_dict_i24(benign_file_path)

	benign_veh_ids = list(benign_sim_dict.keys())

	acc_veh_ids = []
	human_veh_ids = []

	for veh_id in benign_veh_ids:
		if(veh_id[:3]=='acc'): acc_veh_ids.append(veh_id)
		else: human_veh_ids.append(veh_id)

	want_50_percent_GPS = False
	if(want_50_percent_GPS):
		# Get a certain penetration rate of GPS tracked vehicles:
		GPS_penetration_rate = 0.50
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

			temp_veh_data = np.array(benign_sim_dict[veh_id])
			num_sim_points = len(temp_data[:,0])

			sample_start_points = np.random.randint(num_sim_points,size=num_samples_per_veh)

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
			temp_veh_data = np.array(benign_sim_dict[veh_id])
			num_sim_points = len(temp_data[:,0])

			sample_start_points = np.random.randint(num_sim_points,size=num_samples_per_veh)

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
		all_vehicle_ids_measured = benign_veh_ids

		# Extract data to perform training on:
		num_samples_per_veh = 10
		training_data_list = []

		seq_len = 100

		print('Extracting measurements...')

		num_veh_ids_extracted = 10
		num_veh_ids = len(benign_veh_ids)

		for veh_id in all_vehicle_ids_measured:

			temp_veh_data = np.array(benign_sim_dict[veh_id])

			num_sim_points = len(temp_veh_data[:,0])

			if(num_sim_points > seq_len):

				sample_start_points = np.random.randint(num_sim_points-seq_len,size=num_samples_per_veh)
				# Begin by just considering speed, acceleration, and measured spacing:
				
				effective_leader_speed,effective_spacing = get_effective_leader_measurements(benign_sim_dict,veh_id,all_vehicle_ids_measured)


				for i in range(num_samples_per_veh):

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

				num_veh_ids_extracted += 1
				sys.stdout.write('\r'+'Veh ids processed: '+str(num_veh_ids_extracted)+'/'+num_veh_ids)
				 
		print()
		print('Training data found.')

		perform_training = True
		if(perform_training):
			train_X = torch.Tensor(np.array(training_data_list))

			model_file_name = 'cnn_lstm_ae_i24_2400_GPS_100_4feat'

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

			epoch_time = time.time()

			for e in range(n_epoch):
				l = train_epoch(model=model, optimizer=optimizer, dataloader=trainloader)
				train_ls, train_tot = eval_data(model=model, dataloader=trainloader)

				sys.stdout.write('\r'+'Epoch train time: '+str(time.time()-epoch_time))
				epoch_time = time.time()

				avg_loss = 1.0 * train_ls / train_tot
				if e % 10 == 0:
					print("Epoch %d, total loss %f, total predictions %d, avg loss %f" % (e, train_ls, train_tot, avg_loss),
						  datetime.datetime.now())
				if avg_loss < best_loss:
					best_loss = avg_loss
					print("Saving model. Best loss: "+str(best_loss))
					torch.save(model.state_dict(), SAVE_PATH)


	#%% Look at training results:
	SAVE_PATH = '/Users/vanderbilt/Desktop/Research_2020/Traffic_Attack/Anomaly_Detection/ACCIntrusionDetection/models/cnn_lstm_ae_cnn_lstm_ae_ring_1800_GPS_100_4feat.pt'
	model = CNNRecurrentAutoencoder(seq_len, n_features, embedding_dim, cnn_channels, kernel_size, stride, device)
	model = model.load_state_dict(torch.load(SAVE_PATH))

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




