import numpy as np
from load_sim_results import get_sim_data_dict
import sys
sys.path.append('/Users/vanderbilt/Desktop/Research_2020/Traffic_Attack/Anomaly_Detection/ACCIntrusionDetection')

from get_ae_performance import load_ae_model

from utils import sliding_window

import time

#Instance of a congested regime under attack:
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

benign_ids = []
for veh_id in veh_ids:
    if(len(sim_dict[veh_id])>0):
        if veh_id not in mal_ACC_ids:
            benign_ids.append(veh_id)


# time = ACC_data[:,0].astype('float')
# speed = ACC_data[:,4].astype('float')
# accel = ACC_data[:,-11]
# is_mal = ACC_data[:,-2].astype('float')
# rel_pos = ACC_data[:,-6].astype('float')
# lane_number = ACC_data[:,-8].astype('float')
# edge = ACC_data[:,-9]

cnn_lstm_ae_model = load_ae_model('1000epochs','cnn_lstm_ae','high_congestion_speed',device='cpu')

#time how long it takes to run on just the malicious data:
mal_losses = []
begin_time = time.time()
for veh_id in mal_ACC_ids:
    sys.stdout.write('\r'+veh_id)
    veh_data = np.array(sim_dict[veh_id]) 
    speed_vals = veh_data[:,4].astype('float')
    loss_vals = sliding_window(cnn_lstm_ae_model,speed_vals)
    mal_losses.append(loss_vals)

end_time = time.time()

print('Total time: '+str(end_time-begin_time))
#This took about .8 seconds per vehicle, over 150



#%% Extract results from given time-range:
begin_search_time = 1100
end_search_time = 1300
relative_pos_behind = 0
relative_pos_ahead = 600
edge_to_choose = 'Eastbound_3'
lane_to_choose = mal_lane


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

pt.figure()
for veh_id in selected_veh_ids:
    veh_data = selected_data[veh_id]
    if(veh_id in mal_ACC_ids):
        pt.plot(veh_data[0],veh_data[1],'r',linewidth=3)
    else:
        pt.plot(veh_data[0],veh_data[1],'b',linewidth=3)

pt.ylabel('Speed [m/s]')
pt.xlabel('Time [s]')
pt.show()


ben_losses = []
mal_losses = []


pt.figure()
for veh_id in selected_veh_ids:
    veh_data = selected_data[veh_id]
    if(len(veh_data[0])>100):
        if(veh_id in mal_ACC_ids):
            [preds,loss] = sliding_window(cnn_lstm_ae_model,veh_data[1])
            mal_losses.append(loss)
            pt.plot(veh_data[0][-len(loss):],loss,'r',linewidth=3)
        else:
            [preds,loss] = sliding_window(cnn_lstm_ae_model,veh_data[1])
            ben_losses.append(loss)
            pt.plot(veh_data[0][-len(loss):],loss,'b',linewidth=3)

pt.ylabel('Reconstruction loss')
pt.xlabel('Time [s]')
pt.show()
















