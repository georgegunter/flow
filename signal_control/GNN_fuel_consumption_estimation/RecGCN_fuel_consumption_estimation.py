import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN

from tqdm import tqdm

from torch_geometric_temporal.signal import StaticGraphTemporalSignal

from process_sim_data import *


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, time_length):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, time_length, 1)
        self.linear = torch.nn.Linear(time_length, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h


def train_RecGCN(train_dataset,node_features=1,time_length=100,save_model = True):

    model = RecurrentGCN(node_features = node_features, time_length = time_length)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()

    for epoch in tqdm(range(200)):
        cost = 0
        for time, snapshot in enumerate(train_dataset):
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            cost = cost + torch.mean((y_hat-snapshot.y)**2)
        cost = cost / (time+1)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()





    return model,losses


def get_network_data(csv_path,warmup_period,row_num,col_num):

    sim_dict = get_sim_timeseries_all_data(csv_path,warmup_period)

    roads,road_indices = gen_road_segments(col_num, row_num)

    road_segment_connections = get_road_segment_connections(road_indices,row_num,col_num)

    return sim_dict,roads,road_indices,road_segment_connections


def convert_np_array_to_list(x):

    x_list = []

    for i in range(x.shape[0]):
        x_curr = x[i,:]
        x_list.append(x_curr.reshape(len(x_curr),1))

    return x_list



if __name__ == '__main__':


    #########################################################
    #                 Set up training data:                 #
    #########################################################

    # information regarding the simulation:
    csv_path = '/Users/vanderbilt/Desktop/Research_2022/Anti-Flow/examples/data/grid-intersection_20221130-1203011669831381.729383-0_emission.csv'
    warmup_period = 100.0
    row_num = 2
    col_num = 3

    sim_dict,roads,road_indices,road_segment_connections = get_network_data(csv_path,warmup_period,row_num,col_num)

    times,road_counts = get_road_segment_counts(roads,road_indices,sim_dict,warmup_period=warmup_period)

    _,total_fuel_consumption = get_road_segment_fuel_consumption(roads,road_indices,sim_dict,warmup_period=warmup_period)

    edge_index = road_segment_connections #2 x num_edges np array

    num_edges = edge_indices.shape[0]

    edge_weight = np.ones(num_edges,) # Don't have any edge features

    # features: list of num_nodes x num_features
    features = convert_np_array_to_list(road_counts)

    targets = convert_np_array_to_list(total_fuel_consumption)

    num_nodes = features[0].shape[0]


    train_dataset = StaticGraphTemporalSignal(
        edge_index=edge_index,edge_weight=edge_weight,features=features,targets=targets)

    #########################################################
    #                    Train the network:                 #
    #########################################################


    node_features = 1

    time_length = 100 # number of time steps looking over

    model = RecurrentGCN(node_features = node_features, time_length = time_length)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()

    losses_indiv_roads = []

    losses_total_net = []

    for epoch in tqdm(range(200)):
        cost = 0

        loss_indiv_road = 0.0

        for time, snapshot in enumerate(train_dataset):
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr) 

            loss_indiv_road = loss_indiv_road + torch.mean((y_hat-snapshot.y)**2)

            cost = cost + (torch.sum(y_hat)-torch.sum(snapshot.y))**2

        if(epoch%20==0):
            print('Epoch: '+str(epoch)+', Loss: '+str(cost ))




        losses_indiv_roads.append(loss_indiv_road.item())

        losses_total_net.append(cost.item())


        cost = cost / (time+1)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()


    #########################################################
    #                    Test the network:                  #
    #########################################################

    time_length = 100

    eval_model_path = 'models/RecGCN_fuel_estimator_'+str(time_length)+'.pth'

    eval_model = RecurrentGCN(node_features = node_features, time_length = time_length)

    eval_model.load_state_dict(torch.load(eval_model_path))

    eval_model.eval()

    fuel_consumption_estimates = []

    for time, snapshot in enumerate(train_dataset):
            y_hat = eval_model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            fuel_consumption_estimates.append(y_hat.detach().numpy().reshape(num_nodes,))


    fuel_consumption_estimates = np.array(fuel_consumption_estimates)













    