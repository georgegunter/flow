import numpy as np
import matplotlib.pyplot as pt
#%% Load results and process:

no_attack_sims = np.loadtxt('/Users/vanderbilt/Desktop/Research_2020/Traffic_Attack/flow/examples/ring_sim_results_param_sweep_no_attack.csv')
ring_sim_results = np.loadtxt('/Users/vanderbilt/Desktop/Research_2020/Traffic_Attack/flow/examples/ring_sim_results_param_sweep_with_attack.csv')
ring_sim_results = list(ring_sim_results)


pt.figure(figsize=(10,20))
valid_sims = []
for sim in ring_sim_results:
    if(not sim[-1]):
        valid_sims.append(sim)
      
valid_sims = np.array(valid_sims)

#%% Plot
dot_size = 200
attack_dot_size = 600

pt.rcParams.update({'font.size': 30})

pt.figure(figsize=[30,30])
pt.subplot(3,1,1)
pt.scatter(valid_sims[:,2],valid_sims[:,5],c='b',s=dot_size)
pt.scatter(no_attack_sims[:,2],no_attack_sims[:,5],c='r',s=attack_dot_size)
pt.title('Affects of attacks on the ring-road.')
pt.legend(['Attacked scenarios','Benign Scenarios'])
pt.ylabel('Average speed per vehicle [m/s]')
pt.subplot(3,1,2)
pt.scatter(valid_sims[:,2],np.sqrt(valid_sims[:,6]),c='b',s=dot_size)
pt.scatter(no_attack_sims[:,2],np.sqrt(no_attack_sims[:,6]),c='r',s=attack_dot_size)
pt.ylabel('Average speed standard deviation [m/s]')
pt.subplot(3,1,3)
pt.scatter(valid_sims[:,2],valid_sims[:,7],c='b',s=dot_size)
pt.scatter(no_attack_sims[:,2],no_attack_sims[:,7],c='r',s=attack_dot_size)
pt.ylabel('Average Fuel Efficiency [km/l]')
pt.xlabel('Ring Length [m]')
#%% Normalize against the no-attack scenario:
valid_sims = list(valid_sims)
no_attack_sims = list(no_attack_sims)

speed_diffs_normalized = []
speed_std_diffs_normalized = []
fuel_eff_diffs_normalized = []

for sim in valid_sims:
    ring_length = sim[2]
    ave_speed = sim[5]
    speed_std = np.sqrt(sim[6])
    fuel_eff = sim[7]
    
    for s in no_attack_sims:
        if(ring_length == s[2]):
            speed_diffs_normalized.append((ave_speed-s[5]))
            std = np.sqrt(s[6])
            speed_std_diffs_normalized.append(speed_std-std - s[6])
            fuel_eff_diffs_normalized.append((fuel_eff-s[7]))


speed_diffs_normalized = np.array(speed_diffs_normalized)
speed_std_diffs_normalized = np.array(speed_std_diffs_normalized)
fuel_eff_diffs_normalized = np.array(fuel_eff_diffs_normalized)
 
valid_sims = np.array(valid_sims)
no_attack_sims = np.array(no_attack_sims)           
#%% Plot Normalized quantities:

# This doesn't really work very well.

dot_size = 200

pt.figure(figsize=[30,30])
pt.subplot(3,1,1)
pt.scatter(valid_sims[:,2],speed_diffs_normalized,c='b',s=dot_size)
pt.plot([np.min(valid_sims[:,2]),np.max(valid_sims[:,2])],[0,0],'k--')
pt.xlim([np.min(valid_sims[:,2])-20,np.max(valid_sims[:,2])+20])
pt.title('Relative impacts from attacks compared to benign on the ring-road.')
pt.ylabel('Change in average speed [m/s]')
pt.subplot(3,1,2)
pt.scatter(valid_sims[:,2],speed_std_diffs_normalized,c='b',s=dot_size)
pt.plot([np.min(valid_sims[:,2]),np.max(valid_sims[:,2])],[0,0],'k--')
pt.xlim([np.min(valid_sims[:,2])-20,np.max(valid_sims[:,2])+20])
pt.ylabel('Change in speed std [m/s]')
pt.subplot(3,1,3)
pt.scatter(valid_sims[:,2],fuel_eff_diffs_normalized,c='b',s=dot_size)
pt.plot([np.min(valid_sims[:,2]),np.max(valid_sims[:,2])],[0,0],'k--')
pt.xlim([np.min(valid_sims[:,2])-20,np.max(valid_sims[:,2])+20])
pt.ylabel('Change in Fuel Efficiency [km/l]')
pt.xlabel('Ring Length [m]') 
    
#%% Make dict of results

ring_sim_results = list(ring_sim_results)
ring_lengths = list(np.unique(no_attack_sims[:,2]))
ring_sim_dict = dict.fromkeys(ring_lengths)

for ring_length in ring_lengths:
    ring_sim_dict[ring_length] = []
    
for sim in ring_sim_results:
    ring_sim_dict[sim[2]].append(sim)
    
#%% Look at relative differences:
    
ring_length = ring_lengths[0]
temp = ring_sim_dict[ring_length]

speed_diffs = []
speed_std_diffs = []
fuel_eff_diffs = []

for sim in temp:
    ring_length = sim[2]
    ave_speed = sim[5]
    speed_std = np.sqrt(sim[6])
    fuel_eff = sim[7]
    
    for s in no_attack_sims:
        if(ring_length == s[2]):
            speed_diffs.append((ave_speed-s[5]))
            std = np.sqrt(s[6])
            speed_std_diffs.append(speed_std-std - s[6])
            fuel_eff_diffs.append((fuel_eff-s[7]))
    

    
    

    
