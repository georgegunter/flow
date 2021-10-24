import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
import time
import os

from load_sim_results import get_vehicle_data
from load_sim_results import find_time_integral
from load_sim_results import get_sim_results
#%% data location:

csv_files = ['i24_adversarial_sims/Dur_10.0_Mag_-1.0_Inflow_1200_ACCPenetration_0.2_AttackPenetration_0.001_ver_1.csv',
             'i24_adversarial_sims/Dur_5.0_Mag_-0.25_Inflow_1200_ACCPenetration_0.2_AttackPenetration_0.2_ver_1.csv',
             'i24_adversarial_sims/Dur_10.0_Mag_-1.0_Inflow_1200_ACCPenetration_0.2_AttackPenetration_0.2_ver_1.csv',
             'i24_adversarial_sims/Dur_10.0_Mag_-1.0_Inflow_1800_ACCPenetration_0.2_AttackPenetration_0.001_ver_1.csv',
             'i24_adversarial_sims/Dur_5.0_Mag_-0.25_Inflow_1800_ACCPenetration_0.2_AttackPenetration_0.2_ver_1.csv',
             'i24_adversarial_sims/Dur_10.0_Mag_-1.0_Inflow_1800_ACCPenetration_0.2_AttackPenetration_0.2_ver_1.csv',
             'i24_adversarial_sims/Dur_10.0_Mag_-1.0_Inflow_2400_ACCPenetration_0.2_AttackPenetration_0.001_ver_1.csv',
             'i24_adversarial_sims/Dur_5.0_Mag_-0.25_Inflow_2400_ACCPenetration_0.2_AttackPenetration_0.2_ver_1.csv',
             'i24_adversarial_sims/Dur_10.0_Mag_-1.0_Inflow_2400_ACCPenetration_0.2_AttackPenetration_0.2_ver_1.csv']

#%%
def get_departures(vehicle_data):
    departure_times = []
    veh_ids = list(vehicle_data.keys())
    for veh_id in veh_ids:
        if(vehicle_data[veh_id]['edge_id'][-1] == 'Eastbound_8'):
            depart_time = vehicle_data[veh_id]['time'][-1]
            departure_times.append(depart_time)
            
    sorted_departures = np.sort(departure_times)
    cum_departures = np.linspace(1,len(sorted_departures),len(sorted_departures))
            
    return [sorted_departures,cum_departures]

# departure_list = []
# sim_results = []

# for csv_file in csv_files:
#     data_frame = pd.read_csv(csv_file)
#     vehicle_data = get_vehicle_data(data_frame=data_frame,print_progress=True)
#     departures = get_departures(vehicle_data)
#     departure_list.append(departures)
    
 
#%% vehicle data list:
vehicle_data_list = []
for csv_file in csv_files:
    data_frame = pd.read_csv(csv_file)
    vehicle_data = get_vehicle_data(data_frame=data_frame,print_progress=True)
    vehicle_data_list.append(vehicle_data)

#%% Plot departure curves: Low inflow
pt.rcParams.update({'font.size': 35})
pt.figure(figsize=[20,10])

[sorted_departures,cum_departures] = get_departures(vehicle_data_list[0])
pt.plot(sorted_departures,cum_departures)

[sorted_departures,cum_departures] = get_departures(vehicle_data_list[1])
pt.plot(sorted_departures,cum_departures)

[sorted_departures,cum_departures] = get_departures(vehicle_data_list[2])
pt.plot(sorted_departures,cum_departures)

pt.legend(['No attack','Medium Attack','Strong Attack'])
pt.title('Low inflow')
pt.ylabel('Cumulative Departures')
pt.xlabel('Time [s]')

#%% Departure curves: Medium Inflow

pt.figure(figsize=[20,10])

[sorted_departures,cum_departures] = get_departures(vehicle_data_list[3])
pt.plot(sorted_departures,cum_departures)

[sorted_departures,cum_departures] = get_departures(vehicle_data_list[4])
pt.plot(sorted_departures,cum_departures)

[sorted_departures,cum_departures] = get_departures(vehicle_data_list[5])
pt.plot(sorted_departures,cum_departures)

pt.legend(['No attack','Medium Attack','Strong Attack'])
pt.title('Medium inflow')
pt.ylabel('Cumulative Departures')
pt.xlabel('Time [s]')

#%% Departure curves: Medium Inflow

pt.figure(figsize=[20,10])

[sorted_departures,cum_departures] = get_departures(vehicle_data_list[6])
pt.plot(sorted_departures,cum_departures)

[sorted_departures,cum_departures] = get_departures(vehicle_data_list[7])
pt.plot(sorted_departures,cum_departures)

[sorted_departures,cum_departures] = get_departures(vehicle_data_list[8])
pt.plot(sorted_departures,cum_departures)

pt.legend(['No attack','Medium Attack','Strong Attack'])
pt.title('High inflow')
pt.ylabel('Cumulative Departures')
pt.xlabel('Time [s]')    