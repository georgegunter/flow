import numpy as np
import matplotlib.pyplot as plt

import torch

from RecGCN_fuel_consumption_estimation import RecurrentGCN


# testing_csv_path = 'simulation_data/testing_original_networ/low_inflow_testing.csv'

testing_csv_path = 'simulation_data/testing_inflows_original_network/high_inflow_testing_set.csv'

warmup_period = 100.0
row_num = 2
col_num = 3

sim_dict,roads,road_indices,road_segment_connections = get_network_data(testing_csv_path,warmup_period,row_num,col_num)

times,road_counts = get_road_segment_counts(roads,road_indices,sim_dict)

_,total_fuel_consumption = get_road_segment_fuel_consumption(roads,road_indices,sim_dict)

edge_index = road_segment_connections #2 x num_edges np array

num_edges = edge_index.shape[1]

edge_weight = np.ones(num_edges,) # Don't have any edge features

# features: list of num_nodes x num_features
features = convert_np_array_to_list(road_counts)

targets = convert_np_array_to_list(total_fuel_consumption)

num_nodes = len(targets[0])

test_dataset = StaticGraphTemporalSignal(
    edge_index=edge_index,edge_weight=edge_weight,features=features,targets=targets)


# Model trained on MSE from individual roads:

node_features = 1

time_length = 100

model_path = 'Indiv_roads_model.pth'

model = RecurrentGCN(node_features = node_features, time_length = time_length)

model.load_state_dict(torch.load(model_path))

model.eval()

indiv_road_model_total_fuel_consumption_est = []

for time, snapshot in enumerate(test_dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        indiv_road_model_total_fuel_consumption_est.append(y_hat.detach().numpy().reshape(num_nodes,))


# Model trained on SE from total energy:


node_features = 1

time_length = 100

model_path = 'total_net_model.pth'

model = RecurrentGCN(node_features = node_features, time_length = time_length)

model.load_state_dict(torch.load(model_path))

model.eval()

total_net_model_total_fuel_consumption_est = []

for time, snapshot in enumerate(test_dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        total_net_model_total_fuel_consumption_est.append(y_hat.detach().numpy().reshape(num_nodes,))


# PLOTTING:

indiv_model_error = total_fuel_consumption - indiv_road_model_total_fuel_consumption_est
indiv_model_indiv_MSE = np.mean(np.square(indiv_model_error),1)

total_fuel_real = np.sum(total_fuel_consumption,1)
total_fuel_est = np.sum(indiv_road_model_total_fuel_consumption_est,1)

indiv_model_total_SE = np.square(total_fuel_real - total_fuel_est)

total_net_model_error = total_fuel_consumption - total_net_model_total_fuel_consumption_est
total_net_model_indiv_MSE = np.mean(np.square(total_net_model_error),1)

total_fuel_real = np.sum(total_fuel_consumption,1)
total_fuel_est = np.sum(total_net_model_total_fuel_consumption_est,1)

total_net_model_total_SE = np.square(total_fuel_real - total_fuel_est)


fig = plt.figure()

plt.subplot(2,1,1)
plt.plot(indiv_model_indiv_MSE[1000:],linewidth=3,label='Correct model')
plt.plot(total_net_model_indiv_MSE[1000:],linewidth=3,label='Incorrect model')
plt.legend(fontsize=20,loc='center right')
plt.ylabel('MSE on indiv. roads',fontsize=20)

plt.subplot(2,1,2)
plt.plot(total_net_model_total_SE[1000:],linewidth=3,label='Correct model')
plt.plot(indiv_model_total_SE[1000:],linewidth=3,label='Incorrect model')
plt.ylabel('SE on total energy',fontsize=20)

fig.suptitle('High congestion: model comparison',fontsize=20)

plt.show()
