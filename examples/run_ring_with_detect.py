import numpy as np
import sys
sys.path.append('../../Anomaly_Detection/ACCIntrusionDetection/')

import os

import torch

import ray
if(not ray.is_initialized()):
	ray.init(ignore_reinit_error=True)

from filter_track import get_vehicle_data
from filter_track import get_sample_data

import Train_HighD_Detector as THD

from run_ring_adversarial import run_attack_sim
from run_ring_adversarial import iter_run
import load_sim_results as load_sim
from load_sim_results import get_vehicle_data
from load_sim_results import get_sim_results

def train_ae_model(sample_data):

	num_features = len(sample_data[0,:])

	trainset = THD.HighD_Dataset(sample_data)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=False)

	ae_model = THD.AutoEncoder(n_feature=num_features, gpu=0) # Could add in functionality for LSTM

	optimizer = torch.optim.Adam(ae_model.parameters(), lr=1e-3, weight_decay=1e-5)
	n_epochs = 300

	losses = THD.train_model(n_epochs,trainloader,ae_model,optimizer)
	
	[train_preds,train_losses] = THD.eval_inputs(ae_model,sample_data)
	
	return [ae_model,train_preds,train_losses]

def get_sim_sample_data(veh_data,
						data_fields=['speed'],
						window_length=100,
						want_random=True,
						num_random_samples=1,
						want_all_samples=False):
	sample_data_dict = dict.fromkeys(data_fields)
	
	veh_ids = list(veh_data.keys())
	
	for data_field in data_fields:
		sample_data = []
		for veh_id in veh_ids:
			x = veh_data[veh_id][data_field]
			num_samples = len(x)
			num_possible_sample_points = num_samples-window_length
			if(num_samples>window_length):
				if(want_random):
					random_sample_points = np.random.randint(0,num_possible_sample_points,num_random_samples)

					start_points = list(random_sample_points)

					for start_point in start_points:
						sample_data.append(x[start_point:start_point+window_length])
				elif(want_all_samples):
					for i in range(num_possible_sample_points):
						sample_data.append(x[i:i+window_length])
						
		sample_data = np.array(sample_data)
		sample_data_dict[data_field] = sample_data
		
	return sample_data_dict

def get_speeds(veh_data):
	speeds = []
	veh_ids = list(veh_data.keys())
	
	for veh_id in veh_ids:
		speeds.append([veh_id,veh_data[veh_id]['speed']])

	return speeds

def get_headways(veh_data):
	headways = []
	veh_ids = list(veh_data.keys())
	
	for veh_id in veh_ids:
		headways.append([veh_id,veh_data[veh_id]['headway']])

	return headways

def get_speed_diffs(veh_data):
	speed_diffs = []
	veh_ids = list(veh_data.keys())
	
	for veh_id in veh_ids:
		headway = veh_data[veh_id]['headway']
		time = veh_data[veh_id]['time']
		dt = time[1]-time[0]
		speed_diff = np.gradient(headway)/dt
		speed_diffs.append([veh_id,speed_diff])

	return headways

def test_samples(ae_model,testing_data,veh_data,window_length=100):
	testing_outputs = []
	
	for sample in testing_data:
		sample_data = sample[1]
		
		if(len(sample_data)>=window_length):
			veh_id = sample[0]
			[sample_pred,sample_loss] = THD.sliding_window(ae_model,
														   sample_data,
														   feature_num=window_length,
														   want_centering=False)
			max_loss = np.max(sample_loss)
			label = veh_data[veh_id]['is_acc']
			
			testing_outputs.append([label,max_loss,veh_id])
			
	return testing_outputs

# @ray.remote
# def test_samples_ray(ae_model,testing_data,veh_data,window_length=100):
# 	testing_outputs = test_samples(ae_model,testing_data,veh_data,window_length)
# 	return testing_outputs


def check_for_collision(vehicle_data):
	veh_ids = list(vehicle_data.keys())
	has_collision = False
	for veh_id in veh_ids:
		if vehicle_data[veh_id]['has_collision']:
			has_collision = True
			return has_collision

	return has_collision


def get_per_veh_results(vehicle_data):
	number_veh = len(vehicle_data)
	number_ACCs = 0
	for veh_id in list(vehicle_data.keys()):
		is_ACC = len(np.unique(vehicle_data[veh_id]['is_malicious'])) > 1
		if is_ACC:
			number_ACCs += 1
	
	
	fuel_effs = []
	mean_speeds = []
	var_speeds = []
	veh_ids = list(vehicle_data.keys())
	
	has_collision = False
	
	for veh_id in veh_ids:
		speed = vehicle_data[veh_id]['speed']
		mean_speed = np.mean(speed)
		speed_var = np.var(speed)
		mean_speeds.append(mean_speed)
		var_speeds.append(speed_var)
		
		total_fuel = load_sim.find_time_integral(vehicle_data[veh_id]['time'],vehicle_data[veh_id]['fuel'])
		total_distance = vehicle_data[veh_id]['distance'][-1]-vehicle_data[veh_id]['distance'][0]
		fuel_eff = total_distance/total_fuel/1000
		fuel_effs.append(fuel_eff)
		
		if vehicle_data[veh_id]['has_collision']:
			has_collision = True

	return [mean_speeds,var_speeds,fuel_effs,has_collision]

def get_test_outcomes(ae_model,benign_max_loss,vehicle_data,samples):
	test_output = test_samples(ae_model,samples,vehicle_data)
	num_FPs = 0
	num_TPs = 0
	num_FNs = 0
	num_TNs = 0

	for x in test_output:
		true_label = x[0]
		assigned_label = x[1] > benign_max_loss

		if(true_label and assigned_label): num_TPs+=1
		elif(true_label and not assigned_label): num_FNs+=1
		elif(not true_label and assigned_label): num_FPs+=1
		elif(not true_label and not assigned_label): num_TNs+=1

	return [num_FPs,num_TPs,num_FNs,num_TNs]

@ray.remote
def get_test_outcomes_ray(ae_model,benign_max_loss,vehicle_data,samples):
	return get_test_outcomes(ae_model,benign_max_loss,vehicle_data,samples)


def get_detect_result(csv_path,file_name,ae_model,benign_max_loss,data_type='speed'):
	vehicle_data = get_vehicle_data(csv_path,print_progress=True)
	
	sample_data = []
	if(data_type == 'speed'):
		sample_data = get_speeds(vehicle_data)
	elif(data_type == 'headway'):
		sample_data = get_headways(vehicle_data)

	sim_params = load_sim.get_sim_params(file_name)

	[num_FPs,num_TPs,num_FNs,num_TNs] =  get_test_outcomes(ae_model,benign_max_loss,vehicle_data,sample_data)

	results = sim_params

	results.append(num_TPs)
	results.append(num_TNs)
	results.append(num_FPs)
	results.append(num_FNs)

	return results

# @ray.remote
# def get_detect_result_speed_ray(csv_path,file_name,ae_model,benign_max_loss):
# 	return get_detect_result_speed(csv_path,file_name,ae_model,benign_max_loss)

def run_detector_on_list(file_name_list,emission_path,ae_model,benign_max_loss,data_type='speed'):
	detector_result_list = []
	for file_name in file_name_list:
		csv_path = os.path.join(emission_path,file_name)
		test_results = get_detect_result(csv_path,file_name,ae_model,benign_max_loss,data_type=data_type)
		detector_result_list.append(test_results)

	return detector_result_list


# def run_detector_on_list(file_name_list,emission_path,ae_model,benign_max_loss,want_parallel=True,data_type='speed'):
# 	detector_result_list = []
# 	if(want_parallel):
# 		detector_result_ids = []
# 		for file_name in file_name_list:
# 			csv_path = os.path.join(emission_path,file_name)
# 			detector_result_ids.append(get_detect_result_speed_ray.remote(csv_path,file_name,ae_model,benign_max_loss))
# 		detector_result_list = ray.get(detector_result_ids)
# 	else:
# 		for file_name in file_name_list:
# 			csv_path = os.path.join(emission_path,file_name)
# 			test_results = get_detect_result(csv_path,file_name,ae_model,benign_max_loss,data_type=data_type)
# 			detector_result_list.append(test_results)

# 	return detector_result_list



# def get_all_test_outcomes(file_name_list,emission_path,ae_model,benign_max_loss,vehicle_data,samples):





# def get_full_sim_results(vehicle_data,ae_model,benign_max_loss,)






# def train_detector(vehicle_data):

# 	sample_data_dict = get_sim_sample_data(veh_data,want_random=True)
# 	speed_samples = sample_data_dict['speed']
# 	[ae_model,train_preds,train_losses] = train_ae_model(sample_data)


# def get_all_results(csv_path,file_name,ae_model=None,print_loading_progress=True):
# 	vehicle_data = get_vehicle_data(csv_path,file_name,print_progress = print_loading_progress)
	
# 	sim_params = load_sim.get_sim_params(file_name)

# 	[mean_speeds,var_speeds,fuel_effs,has_collision] = get_per_veh_results(vehicle_data)

# 	if(ae_model is not None):
