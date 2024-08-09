import numpy as np
import csv
import os
import time
import ray


from tqdm import tqdm

'''
Functions for loading in timeseries data:
'''


def get_vehicle_data(csv_path,warmup_period=0.0,want_print_finished_loading=True):
    row_num = 1
    curr_veh_id = 'id'
    sim_dict = {}
    curr_veh_data = []

    begin_time = time.time()

    with open(csv_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        id_index = 0
        time_index = 0
        speed_index = 0
        headway_index = 0
        relvel_index = 0
        edge_index = 0
        pos_index = 0

        row1 = next(csvreader)
        num_entries = len(row1)
        while(row1[id_index]!='id' and id_index<num_entries):id_index +=1
        while(row1[edge_index]!='edge_id' and edge_index<num_entries):edge_index +=1
        while(row1[time_index]!='time' and time_index<num_entries):time_index +=1
        while(row1[speed_index]!='speed' and speed_index<num_entries):speed_index +=1
        while(row1[headway_index]!='headway' and headway_index<num_entries):headway_index +=1
        while(row1[relvel_index]!='leader_rel_speed' and relvel_index<num_entries):relvel_index +=1

        for row in csvreader:
            if(row_num > 1):
                # Don't read header
                if(curr_veh_id != row[id_index]):
                    #Add in new data to the dictionary:
                    
                    #Store old data:
                    if(len(curr_veh_data)>0):
                        sim_dict[curr_veh_id] = curr_veh_data
                    #Rest where data is being stashed:
                    curr_veh_data = []
                    curr_veh_id = row[id_index] # Set new veh id
                    #Allocate space for storing:
                    # sim_dict[curr_veh_id] = []

                curr_veh_id = row[id_index]
                sim_time = float(row[time_index])
                edge = row[edge_index]
                if(sim_time > warmup_period):
                    curr_veh_data.append(row)
            row_num += 1

        #Add the very last vehicle's information:
        sim_dict[curr_veh_id] = curr_veh_data
        end_time = time.time()
        if(want_print_finished_loading):
            print('Data loaded, total time: '+str(end_time-begin_time))
        

    return sim_dict




def get_trajectory_timeseries(csv_path,warmup_period=0.0,want_print_finished_loading=True):
    row_num = 1
    curr_veh_id = 'id'
    sim_dict = {}
    curr_veh_data = []

    begin_time = time.time()

    with open(csv_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        id_index = 0
        time_index = 0
        speed_index = 0
        headway_index = 0
        relvel_index = 0
        edge_index = 0
        pos_index = 0

        row1 = next(csvreader)
        num_entries = len(row1)
        while(row1[id_index]!='id' and id_index<num_entries):id_index +=1
        while(row1[edge_index]!='edge_id' and edge_index<num_entries):edge_index +=1
        while(row1[time_index]!='time' and time_index<num_entries):time_index +=1
        while(row1[speed_index]!='speed' and speed_index<num_entries):speed_index +=1
        while(row1[headway_index]!='headway' and headway_index<num_entries):headway_index +=1
        while(row1[relvel_index]!='leader_rel_speed' and relvel_index<num_entries):relvel_index +=1

        for row in csvreader:
            if(row_num > 1):
                # Don't read header
                if(curr_veh_id != row[id_index]):
                    #Add in new data to the dictionary:
                    
                    #Store old data:
                    if(len(curr_veh_data)>0):
                        sim_dict[curr_veh_id] = np.array(curr_veh_data).astype(float)
                    #Rest where data is being stashed:
                    curr_veh_data = []
                    curr_veh_id = row[id_index] # Set new veh id
                    #Allocate space for storing:
                    # sim_dict[curr_veh_id] = []

                curr_veh_id = row[id_index]
                sim_time = float(row[time_index])
                edge = row[edge_index]
                if(sim_time > warmup_period):
                    # data = [time,speed,headway,leader_rel_speed]

                    # Check what was filled in if missing a leader:
                    s = float(row[headway_index])
                    dv = float(row[relvel_index])
                    v = float(row[speed_index])
                    t = float(row[time_index])

                    data = [t,v,s,dv]
                    curr_veh_data.append(data)
            row_num += 1

        #Add the very last vehicle's information:
        sim_dict[curr_veh_id] = np.array(curr_veh_data).astype(float)
        end_time = time.time()
        if(want_print_finished_loading):
            print('Data loaded, total time: '+str(end_time-begin_time))
        

    return sim_dict


def get_gaussian_kernel_filter(times_list,
    positions_list,
    speeds_list):


    min_time = np.inf
    max_time = 0.0

    for times in times_list:
        min_time = np.min([np.min(times),min_time])

        max_time = np.max([np.max(times),max_time])


    min_position = np.inf
    max_position = 0.0

    for positions in positions_list:
        min_position = np.min([np.min(positions),min_position])

        max_position = np.max([np.max(positions),max_position])


    t_begin = min_time
    t_end = max_time

    dt = times_list[0][1] - times_list[0][0]

    wx = 10.0 # width of kernel in x
    wt = 10.0 # width of kernel in t

    mt = np.arange(t_begin,t_end,5.0)

    mx = np.arange(min_position,max_position,1.0)

    [MT,MX] = np.meshgrid(mt,mx)

    MRho = MT*0 #Density
    MQ = MT*0 #Flow

    for i in tqdm(range(len(times_list))):
        times = times_list[i]
        positions = positions_list[i]
        speeds = speeds_list[i]

        num_samples = len(times)

        for j in range(num_samples):
            x = positions[j]
            t = times[j]
            v = speeds[j]

            dist = np.abs(MX - x)

            G = np.exp(-np.square(dist/wx) - np.square((MT - t)/wt))/(np.pi*wx*wt)*dt

            MRho = MRho + G

            MQ = MQ + G*v

    return MRho,MQ

