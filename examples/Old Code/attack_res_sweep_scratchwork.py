
#%% plotting function:

def plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration,markersize=20,alpha=0.3,linewidth=5.0):    
    x_vals = []
    y_vals = []
    min_y_vals = []
    max_y_vals = []
    std_y_vals    
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
            
    temp = np.argsort(x_vals)        
    x_vals_sorted = np.zeros_like(x_vals)
    y_vals_sorted = np.zeros_like(y_vals)
    min_y_vals_sorted = np.zeros_like(y_vals)
    max_y_vals_sorted = np.zeros_like(y_vals)
    
    for i in range(len(temp)):
        x_ind = temp[i]
        x_vals_sorted[i] = x_vals[x_ind]
        y_vals_sorted[i] = y_vals[x_ind]
        min_y_vals_sorted[i] = min_y_vals[x_ind]
        max_y_vals_sorted[i] = max_y_vals[x_ind]
    
            
    axis.plot(x_vals_sorted,y_vals_sorted,plot_string,linewidth=linewidth,markersize=markersize)
    axis.fill_between(x_vals_sorted,min_y_vals_sorted,max_y_vals_sorted,
                    facecolor=face_color, alpha=alpha)
    
#%%
fig, ax = pt.subplots(figsize=[20,10])

axis = ax
param_index = 3    
res_index=-3
inflow_rate=2400
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
plot_sim_res_shaded(axis,attack_files,param_index,res_index,inflow_rate,attack_penetration)    

#%% Plot across all results:

   
    fig, axes = pt.subplots(3,3,figsize=[40,30])

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
                        want_y_labels=True,ylims=[2850,2900])
    
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
    axis=axis=axes[1][2]

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
    res_index = -2
    
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
    axis=axis=axes[2][2]

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
        
     
        
        
        




    