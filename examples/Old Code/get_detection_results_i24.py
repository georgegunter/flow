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

# from run_ring_adversarial import run_attack_sim
# from run_ring_adversarial import iter_run
import load_sim_results as load_sim
from load_sim_results import get_vehicle_data
from load_sim_results import get_sim_results

#%% useful functions for training and testing:

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

def get_sim_training_data(veh_data,
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

def get_test_results(ae_model,testing_data,veh_data,window_length=100):
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



#%% Train for first scenario:
print('Beginning training process...')   
low_training_veh_data = vehicle_data_list[0]
low_training_data = get_sim_training_data(low_training_veh_data,num_random_samples=10)
low_training_data = low_training_data['speed']
[ae_model,train_preds,train_losses] = train_ae_model(low_training_data)
low_ae_model = ae_model
print('Finished training model.')

#%% get testing results:

print('Testing learned model')
low_test_data_low = get_speeds(low_training_veh_data)
low_test_results_low = get_test_results(ae_model=low_ae_model, testing_data=low_test_data_low, veh_data=low_training_veh_data)


low_test_veh_data_medium = vehicle_data_list[1]
low_test_data_medium = get_speeds(low_test_veh_data_medium)
low_test_results_medium = get_test_results(low_ae_model, low_test_data_medium, low_test_veh_data_medium)
 
low_test_veh_data_high = vehicle_data_list[2]
low_test_data_high = get_speeds(low_test_veh_data_high)
low_test_results_high = get_test_results(low_ae_model, low_test_data_high, low_test_veh_data_high)

print('Testing complete.')

#%% Train for second scenario:
print('Beginning training process...')   
med_training_veh_data = vehicle_data_list[3]
med_training_data = get_sim_training_data(med_training_veh_data,num_random_samples=10)
med_training_data = med_training_data['speed']
[ae_model,train_preds,train_losses] = train_ae_model(med_training_data)
med_ae_model = ae_model
print('Finished training model.')

print('Testing learned model')
med_test_data_low = get_speeds(med_training_veh_data)
med_test_results_low = get_test_results(ae_model=med_ae_model, testing_data=med_test_data_low, veh_data=med_training_veh_data)


med_test_veh_data_medium = vehicle_data_list[4]
med_test_data_medium = get_speeds(med_test_veh_data_medium)
med_test_results_medium = get_test_results(med_ae_model, med_test_data_medium, med_test_veh_data_medium)
 
med_test_veh_data_high = vehicle_data_list[5]
med_test_data_high = get_speeds(med_test_veh_data_high)
med_test_results_high = get_test_results(med_ae_model, med_test_data_high, med_test_veh_data_high)

print('Testing complete.')

#%% Train for third scenario:
print('Beginning training process...')   
high_training_veh_data = vehicle_data_list[6]
high_training_data = get_sim_training_data(high_training_veh_data,num_random_samples=10)
high_training_data = high_training_data['speed']
[ae_model,train_preds,train_losses] = train_ae_model(high_training_data)
high_ae_model = ae_model
print('Finished training model.')

print('Testing learned model')
high_test_data_low = get_speeds(high_training_veh_data)
high_test_results_low = get_test_results(ae_model=high_ae_model, testing_data=high_test_data_low, veh_data=high_training_veh_data)


high_test_veh_data_medium = vehicle_data_list[7]
high_test_data_medium = get_speeds(high_test_veh_data_medium)
high_test_results_medium = get_test_results(high_ae_model, high_test_data_medium, high_test_veh_data_medium)
 
high_test_veh_data_high = vehicle_data_list[8]
high_test_data_high = get_speeds(high_test_veh_data_high)
high_test_results_high = get_test_results(high_ae_model, high_test_data_high, high_test_veh_data_high)

print('Testing complete.')

#%%

def get_classifier_stats(test_results,max_loss):
    num_FPs = 0
    num_FNs = 0
    num_TPs = 0
    num_TNs = 0
    
    for x in test_results:
        y = x[0] #True label
        f_x = x[1] > max_loss #Assigned label
        
        if(y and f_x): num_TPs += 1
        elif(not y and not f_x): num_TNs += 1
        elif(not y and not f_x): num_TNs += 1
        elif(not y and f_x): num_FPs += 1
        elif(y and not f_x): num_FNs += 1
        
    return [num_FPs,num_FNs,num_TPs,num_TNs]

def get_losses(test_results):
    losses = []
    for x in test_results:
        losses.append(x[1])
    return np.array(losses)

def get_labels(test_results):
    labels = []
    for x in test_results:
        labels.append(x[0])
    return np.array(labels)
    
#%% 

low_losses_none = get_losses(low_test_results_low)
low_max_loss = np.max(low_losses_none)
low_classifier_stats_none = get_classifier_stats(low_test_results_low,low_max_loss)
low_classifier_stats_weak = get_classifier_stats(low_test_results_medium,low_max_loss)
low_classifier_stats_strong = get_classifier_stats(low_test_results_high,low_max_loss)

med_losses_none = get_losses(med_test_results_low)
med_max_loss = np.max(med_losses_none)
med_classifier_stats_none = get_classifier_stats(med_test_results_low,med_max_loss)
med_classifier_stats_weak = get_classifier_stats(med_test_results_medium,med_max_loss)
med_classifier_stats_strong = get_classifier_stats(med_test_results_high,med_max_loss)

high_losses_none = get_losses(high_test_results_low)
high_max_loss = np.max(med_losses_none)
high_classifier_stats_none = get_classifier_stats(high_test_results_low,high_max_loss)
high_classifier_stats_weak = get_classifier_stats(high_test_results_medium,high_max_loss)
high_classifier_stats_strong = get_classifier_stats(high_test_results_high,high_max_loss)

print(low_classifier_stats_none)
print(low_classifier_stats_weak)
print(low_classifier_stats_strong) 
print(med_classifier_stats_none)
print(med_classifier_stats_weak)
print(med_classifier_stats_strong) 
print(high_classifier_stats_none)
print(high_classifier_stats_weak)
print(high_classifier_stats_strong) 














