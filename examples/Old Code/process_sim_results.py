import numpy as np
import matplotlib.pyplot as pt
import os

def get_all_sim_results(csv_repo_path='/Users/vanderbilt/Desktop/Research_2020/Traffic_Attack/flow/examples/i24_adversarial_sims/results_csv_repo'):
    files = os.listdir(csv_repo_path)
    unique_file_names = []

    for file in files:
        if(file[-3:] == 'csv'):
            i=0
            while((file[i:i+2] != '_v') and (i < len(file)-1)): i+=1
            file_name = file[:i]
            if(file_name not in unique_file_names): unique_file_names.append(file_name)

    sim_results_dict = dict.fromkeys(unique_file_names)
    for file_name in unique_file_names:
        sim_results_dict[file_name] = []

    for file in files:
        if(file[-3:] == 'csv'):
            i=0
            while((file[i:i+2] != '_v') and (i < len(file)-1)): i+=1
            file_name = file[:i]

            csv_file_path = os.path.join(csv_repo_path,file)
            sim_results = np.loadtxt(csv_file_path)

            sim_results_dict[file_name].append(sim_results)

    for file in unique_file_names:
        sim_results_dict[file_name] = np.array(sim_results_dict[file_name])

    return sim_results_dict

def plot_all_sim_res(sim_results_dict,file_name):


