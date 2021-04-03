import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
import time
import os

from load_sim_results import get_vehicle_data
from load_sim_results import find_time_integral
#%% Load data:

inflow_rate = '1200'

start_time = time.time()

data_frame_no_attack = pd.read_csv('i24_adversarial_sims/Dur_10.0_Mag_-1.0_Inflow_'+inflow_rate+'_ACCPenetration_0.2_AttackPenetration_0.001_ver_1.csv')
data_frame_medium_attack = pd.read_csv('i24_adversarial_sims/Dur_5.0_Mag_-0.25_Inflow_'+inflow_rate+'_ACCPenetration_0.2_AttackPenetration_0.2_ver_1.csv')
data_frame_full_attack = pd.read_csv('i24_adversarial_sims/Dur_10.0_Mag_-1.0_Inflow_'+inflow_rate+'_ACCPenetration_0.2_AttackPenetration_0.2_ver_1.csv')




#%%
start_time = time.time()
vehicle_data_no_attack = get_vehicle_data(data_frame=data_frame_no_attack,print_progress=True)
vehicle_data_medium_attack = get_vehicle_data(data_frame=data_frame_medium_attack,print_progress=True)
vehicle_data_full_attack = get_vehicle_data(data_frame=data_frame_full_attack,print_progress=True)

end_time = time.time()
print('Load time: '+str(end_time-start_time))

#%%
fuel_effs_no_attack = []
mean_speeds_no_attacks = []
var_speeds_no_attacks = [] 
veh_ids_no_attack = list(vehicle_data_no_attack.keys())
for veh_id in veh_ids_no_attack:
    
    is_acc = vehicle_data_no_attack[veh_id]['is_acc']
    if not is_acc:
        speed = vehicle_data_no_attack[veh_id]['speed']
        mean_speed = np.mean(speed)
        speed_var = np.var(speed)
        mean_speeds_no_attacks.append(mean_speed)
        var_speeds_no_attacks.append(speed_var)
        
        total_fuel = find_time_integral(vehicle_data_no_attack[veh_id]['time'],vehicle_data_no_attack[veh_id]['fuel'])
        total_distance = vehicle_data_no_attack[veh_id]['distance'][-1]-vehicle_data_no_attack[veh_id]['distance'][0]
        fuel_eff = total_distance/total_fuel/1000
        fuel_effs_no_attack.append(fuel_eff)
    
    
fuel_effs_medium_attack = []
mean_speeds_medium_attacks = []
var_speeds_medium_attacks = [] 
veh_ids_medium_attack = list(vehicle_data_medium_attack.keys())
for veh_id in veh_ids_medium_attack:
    
    is_acc = vehicle_data_medium_attack[veh_id]['is_acc']
    if not is_acc:
        speed = vehicle_data_medium_attack[veh_id]['speed']
        mean_speed = np.mean(speed)
        speed_var = np.var(speed)
        mean_speeds_medium_attacks.append(mean_speed)
        var_speeds_medium_attacks.append(speed_var)
        
        total_fuel = find_time_integral(vehicle_data_medium_attack[veh_id]['time'],vehicle_data_medium_attack[veh_id]['fuel'])
        total_distance = vehicle_data_medium_attack[veh_id]['distance'][-1]-vehicle_data_medium_attack[veh_id]['distance'][0]
        fuel_eff = total_distance/total_fuel/1000
        fuel_effs_medium_attack.append(fuel_eff)
    
fuel_effs_full_attack = []
mean_speeds_full_attacks = []
var_speeds_full_attacks = [] 
veh_ids_full_attack = list(vehicle_data_full_attack.keys())
for veh_id in veh_ids_full_attack:
    
    is_acc = vehicle_data_full_attack[veh_id]['is_acc']
    if not is_acc:
        speed = vehicle_data_full_attack[veh_id]['speed']
        mean_speed = np.mean(speed)
        speed_var = np.var(speed)
        mean_speeds_full_attacks.append(mean_speed)
        var_speeds_full_attacks.append(speed_var)
        
        total_fuel = find_time_integral(vehicle_data_full_attack[veh_id]['time'],vehicle_data_full_attack[veh_id]['fuel'])
        total_distance = vehicle_data_full_attack[veh_id]['distance'][-1]-vehicle_data_full_attack[veh_id]['distance'][0]
        fuel_eff = total_distance/total_fuel/1000
        fuel_effs_full_attack.append(fuel_eff)
    
std_speeds_no_attacks = np.sqrt(var_speeds_no_attacks)
std_speeds_medium_attacks = np.sqrt(var_speeds_medium_attacks)
std_speeds_full_attacks = np.sqrt(var_speeds_full_attacks)    
#%% Plotting for the mean speeds:

pt.rcParams.update({'font.size': 35})
    
pt.figure(figsize=[25,20])
pt.subplot(2,1,1)
pt.hist(mean_speeds_no_attacks,100,density=True,histtype='step',color='green',LineWidth=5)
pt.hist(mean_speeds_medium_attacks,100,density=True,histtype='step',color='blue',LineWidth=5)
pt.hist(mean_speeds_full_attacks,100,density=True,histtype='step',color='red',LineWidth=5)
pt.legend(['No attacks','Medium attacks','Strong attacks'],loc=2)
pt.xlabel('Mean speed [m/s]')
pt.title('Network Inflow: '+inflow_rate+' veh/hr')
    
pt.subplot(2,1,2)
pt.hist(std_speeds_no_attacks,100,density=True,histtype='step',color='green',LineWidth=5)
pt.hist(std_speeds_medium_attacks,100,density=True,histtype='step',color='blue',LineWidth=5)
pt.hist(std_speeds_full_attacks,100,density=True,histtype='step',color='red',LineWidth=5)
pt.xlabel('Speed Standard Deviation [m/s]')

#%% Just the first case :
    











    
    