import numpy as np
import matplotlib.pyplot as pt
from process_sim_results import get_all_sim_results


#%% Useful functions
def get_sim_params(file):
    params = []
    num_chars = len(file)
    i=0
    while(file[i] != '_'): i+= 1
    i += 1
    j = i
    while(file[j] != '_'): j+= 1
    params.append(float(file[i:j]))
    i = j+1
    
    while(i < num_chars-4):
        while(file[i] != '_'): i+= 1
        i += 1
        j = i
        while(file[j] != '_' and j<num_chars-1): j+= 1
        if(j==num_chars-1):params.append(float(file[i:]))
        else:params.append(float(file[i:j]))
        i = j+1
    		
    return params

def plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration,
                        want_std_fill=True,markersize=25,alpha=0.3,linewidth=7.0,fontsize=35,ticksize=30,
                        want_y_labels=False,want_x_labels=False,xlims=None,ylims=None,want_grids=True):    
    x_vals = []
    y_vals = []
    min_y_vals = []
    max_y_vals = []
    std_y_vals = []
    
    
    plot_string = ''
    if(attack_penetration == 0.1):plot_string = '.-'
    else:plot_string = '*-'
        
    face_color = ''
    p_temp = get_sim_params(attack_files[0])
    
    if(p_temp[0]==10.0):
        plot_string = 'r'+plot_string
        face_color = 'red'
    elif(p_temp[0]==5.0):
        plot_string = 'b'+plot_string
        face_color = 'blue'
    elif(p_temp[0]==0.1):
        plot_string = 'g.-'
        face_color = 'green'
    
    for file in attack_files:
        params = get_sim_params(file)
        res = all_sim_results[file]
        
        if(params[2]==inflow_rate and params[4] == attack_penetration):
            print(file)
            x_vals.append(params[param_index])
            y_vals.append(np.mean(res[:,res_index]))
            min_y_vals.append(np.min(res[:,res_index]))
            max_y_vals.append(np.max(res[:,res_index]))
            std_y_vals.append(np.std(res[:,res_index]))
            
    temp = np.argsort(x_vals)        
    x_vals_sorted = np.zeros_like(x_vals)
    y_vals_sorted = np.zeros_like(y_vals)
    min_y_vals_sorted = np.zeros_like(y_vals)
    max_y_vals_sorted = np.zeros_like(y_vals)
    std_y_vals_sorted = np.zeros_like(y_vals)
    
    for i in range(len(temp)):
        x_ind = temp[i]
        x_vals_sorted[i] = x_vals[x_ind]
        y_vals_sorted[i] = y_vals[x_ind]
        min_y_vals_sorted[i] = min_y_vals[x_ind]
        max_y_vals_sorted[i] = max_y_vals[x_ind]
        std_y_vals_sorted[i] = std_y_vals[x_ind]
    
            
    axis.plot(x_vals_sorted,y_vals_sorted,plot_string,linewidth=linewidth,markersize=markersize)
    
    if(want_std_fill):
        axis.fill_between(x_vals_sorted,y_vals_sorted-std_y_vals_sorted,y_vals_sorted+std_y_vals_sorted,
                    facecolor=face_color, alpha=alpha)
    else:    
        axis.fill_between(x_vals_sorted,min_y_vals_sorted,max_y_vals_sorted,
                    facecolor=face_color, alpha=alpha)

    if(want_y_labels):
        
        ylabel=''
        if(res_index==-3):ylabel='Outflow [veh]'
        elif(res_index==-2):ylabel='Mean traffic speed [m/s]'
        elif(res_index==-1):ylabel='Mean traffic speed variance [m/s]'    
    
        axis.set_ylabel(ylabel,fontsize=fontsize)
        
    if(want_x_labels):
        axis.set_xlabel('ACC penetration rate',fontsize=fontsize)
    
    if(xlims is not None):
        axis.set_xlim(xlims)
    if(ylims is not None):
        axis.set_ylim(ylims)

    if(want_grids):axis.grid()
    
    axis.tick_params(axis='x', labelsize=ticksize)
    axis.tick_params(axis='y', labelsize=ticksize)
    
    
        
    

#%% 
if __name__ == "__main__":
#%%
    all_sim_results = get_all_sim_results()
    sim_files = list(all_sim_results.keys())
    for file in sim_files:
        all_sim_results[file] = np.array(all_sim_results[file])
        
    strong_attack_files = []
    weak_attack_files = []
    no_attack_files = []

    for file in sim_files:
        if('Dur_10.0_Mag_-1.0' in file):
            strong_attack_files.append(file)
        if('Dur_5.0_Mag_-0.25' in file):
            weak_attack_files.append(file)
        if('Dur_0.1_Mag_-0.1' in file):
            no_attack_files.append(file)
            
#%% Plot across all results:

   
    fig, axes = pt.subplots(3,3,figsize=[70,30])

    param_index = 3

    # Inflows:    
    res_index = -3
    
    axis = axes[0][0]    
    inflow_rate = 2400
    
    attack_penetration = 0.2
    attack_files = no_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_files = weak_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_files = strong_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_penetration = 0.1
    attack_files = weak_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_files = strong_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration,
                        want_y_labels=True,ylims=[3800,5200])
    
    axis.set_title('High congestion',fontsize=60)
    
    inflow_rate = 1800
    axis=axes[0][1]
    
    attack_penetration = 0.2
    attack_files = no_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_files = weak_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_files = strong_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_penetration = 0.1
    attack_files = weak_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_files = strong_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration,
                        want_y_labels=True,ylims=[3800,4300])

    axis.set_title('Medium congestion',fontsize=60)
    
    inflow_rate = 1200
    axis=axis=axes[0][2]

    attack_penetration = 0.2
    attack_files = no_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_files = weak_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_files = strong_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_penetration = 0.1
    attack_files = weak_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_files = strong_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration,
                        want_y_labels=True,ylims=[2700,3000])
    
    axis.set_title('Low congestion',fontsize=60)
    #Mean traffic speed
    res_index = -2
    
    axis = axes[1][0]    
    inflow_rate = 2400
    
    attack_penetration = 0.2
    attack_files = no_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_files = weak_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_files = strong_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_penetration = 0.1
    attack_files = weak_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_files = strong_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration,
                        want_y_labels=True)
    

    
    inflow_rate = 1800
    axis=axes[1][1]
    
    attack_penetration = 0.2
    attack_files = no_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_files = weak_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_files = strong_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_penetration = 0.1
    attack_files = weak_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_files = strong_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration,
                        want_y_labels=True)


    inflow_rate = 1200
    axis=axes[1][2]

    attack_penetration = 0.2
    attack_files = no_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_files = weak_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_files = strong_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_penetration = 0.1
    attack_files = weak_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_files = strong_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration,
                        want_y_labels=True)
    
    
    #Traffic speed variance:
    res_index = -1
    
    axis = axes[2][0]    
    inflow_rate = 2400
    
    attack_penetration = 0.2
    attack_files = no_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_files = weak_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_files = strong_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_penetration = 0.1
    attack_files = weak_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_files = strong_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration,
                        want_y_labels=True,want_x_labels=True)
    

    
    inflow_rate = 1800
    axis=axes[2][1]
    
    attack_penetration = 0.2
    attack_files = no_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_files = weak_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_files = strong_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_penetration = 0.1
    attack_files = weak_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_files = strong_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration,
                        want_y_labels=True,want_x_labels=True)


    inflow_rate = 1200
    axis=axes[2][2]

    attack_penetration = 0.2
    attack_files = no_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_files = weak_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_files = strong_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_penetration = 0.1
    attack_files = weak_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    
    attack_files = strong_attack_files        
    plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration,
                        want_y_labels=True,want_x_labels=True)
    
 
