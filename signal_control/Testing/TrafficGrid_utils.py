from flow.controllers import GridRouter
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams
from flow.core.params import SumoCarFollowingParams
from flow.core.params import InFlows
from flow.envs.traffic_light_grid import ADDITIONAL_ENV_PARAMS
from flow.envs import TrafficLightGridGenericObserver
from flow.networks import TrafficLightGridNetwork

from flow.core.experiment import Experiment

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces import Tuple

import numpy as np



def gen_edges(col_num, row_num):
    """Generate the names of the outer edges in the grid network.

    Parameters
    ----------
    col_num : int
        number of columns in the grid
    row_num : int
        number of rows in the grid

    Returns
    -------
    list of str
        names of all the outer edges
    """
    edges = []

    # build the left and then the right edges
    for i in range(col_num):
        edges += ['left' + str(row_num) + '_' + str(i)]
        edges += ['right' + '0' + '_' + str(i)]

    # build the bottom and then top edges
    for i in range(row_num):
        edges += ['bot' + str(i) + '_' + '0']
        edges += ['top' + str(i) + '_' + str(col_num)]

    return edges


def get_flow_params(col_num, row_num, additional_net_params,main_flow,arterial_flow):
    """Define the network and initial params in the presence of inflows.

    Parameters
    ----------
    col_num : int
        number of columns in the grid
    row_num : int
        number of rows in the grid
    additional_net_params : dict
        network-specific parameters that are unique to the grid

    Returns
    -------
    flow.core.params.InitialConfig
        parameters specifying the initial configuration of vehicles in the
        network
    flow.core.params.NetParams
        network-specific parameters used to generate the network
    """
    initial = InitialConfig(
        spacing='custom', lanes_distribution=float('inf'), shuffle=True)

    inflow = InFlows()
    outer_edges = gen_edges(col_num, row_num)
    for i in range(len(outer_edges)):
        edge_id = outer_edges[i]
        if('left' in edge_id):
            inflow.add(
                veh_type='human',
                edge=edge_id,
                vehs_per_hour=main_flow,
                depart_lane='free',
                depart_speed=20)
        else:
            inflow.add(
                veh_type='human',
                edge=edge_id,
                vehs_per_hour=arterial_flow,
                depart_lane='free',
                depart_speed=20)

    net = NetParams(
        inflows=inflow,
        additional_params=additional_net_params)

    return initial, net


def get_non_flow_params(enter_speed, add_net_params):
    """Define the network and initial params in the absence of inflows.

    Note that when a vehicle leaves a network in this case, it is immediately
    returns to the start of the row/column it was traversing, and in the same
    direction as it was before.

    Parameters
    ----------
    enter_speed : float
        initial speed of vehicles as they enter the network.
    add_net_params: dict
        additional network-specific parameters (unique to the grid)

    Returns
    -------
    flow.core.params.InitialConfig
        parameters specifying the initial configuration of vehicles in the
        network
    flow.core.params.NetParams
        network-specific parameters used to generate the network
    """
    additional_init_params = {'enter_speed': enter_speed}
    initial = InitialConfig(
        spacing='custom', additional_params=additional_init_params)
    net = NetParams(additional_params=add_net_params)

    return initial, net



def observer_counts(env):
    edge_ids = env.k.network.get_edge_list()

    counts = dict.fromkeys(edge_ids)
    
    for edge_id in edge_ids:
        # How many veh_ids are on each edge is the count:
        ids_by_edge = env.k.vehicle.get_ids_by_edge(edge_id)
        counts[edge_id] = len(ids_by_edge)
    
    edge_counts = []
    for edge_id in edge_ids:
        edge_counts.append(counts[edge_id])
    
    return np.array(edge_counts)



def observer_traffic_light_state(env):
    traffic_light_ids = env.k.traffic_light.get_ids()
    traffic_light_states = []
    for tl_id in traffic_light_ids:
        traffic_light_states.append(env.k.traffic_light.get_state(tl_id))
    return traffic_light_states


def observer_ave_veh_speed_counts_tl_states(env):
    
    edge_ids = env.k.network.get_edge_list()
    
    vehicular_speeds_by_edge = {}
    
    for edge_id in edge_ids:
        # How many veh_ids are on each edge is the count:
        ids_by_edge = env.k.vehicle.get_ids_by_edge(edge_id)
        
        veh_speeds = []
        
        if(len(ids_by_edge) == 0):
            veh_speeds.append(30.0)
        else:
            for veh_id in ids_by_edge:
                current_veh_speed = env.k.vehicle.get_speed(veh_id)
                veh_speeds.append(current_veh_speed)
                
        vehicular_speeds_by_edge[edge_id] = np.mean(veh_speeds)
        
    ave_veh_speeds = []
    for edge_id in edge_ids:
        ave_veh_speeds.append(vehicular_speeds_by_edge[edge_id])
        
    
    edge_ids = env.k.network.get_edge_list()

    counts = dict.fromkeys(edge_ids)
    
    for edge_id in edge_ids:
        # How many veh_ids are on each edge is the count:
        ids_by_edge = env.k.vehicle.get_ids_by_edge(edge_id)
        counts[edge_id] = len(ids_by_edge)
    
    edge_counts = []
    for edge_id in edge_ids:
        edge_counts.append(counts[edge_id])
    
    traffic_light_ids = env.k.traffic_light.get_ids()
    traffic_light_states = []
    for tl_id in traffic_light_ids:
        traffic_light_states.append(env.k.traffic_light.get_state(tl_id))
        

    sim_time = env.step_counter*env.sim_step
        
    return [edge_counts,ave_veh_speeds,traffic_light_states,sim_time]





def get_ave_veh_speed_counts_tl_states_timeseries(states):
    all_counts = []
    all_speeds = []
    all_tl_states = []

    for state in states:
        counts = state[0]
        speeds = state[1]
        tl_states = state[2]

        all_counts.append(counts)
        all_speeds.append(speeds)
        all_tl_states.append(tl_states)
        
    return all_counts,all_speeds,all_tl_states


def numerical_tl_states(tl_states):
    tl_states_numerical = []
    
    state_1 = 'rGrG'
    state_2 = 'ryry'
    state_3 = 'GrGr'
    state_4 = 'yryr'
    
    for tl_state in tl_states:
        curr_num_state = []
        for i in range(len(tl_state)):
            if(tl_state[i] == state_1):
                curr_num_state.append(1)
            elif(tl_state[i] == state_2):
                curr_num_state.append(2)
            elif(tl_state[i] == state_3):
                curr_num_state.append(3)
            elif(tl_state[i] == state_4):
                curr_num_state.append(4)
                
        tl_states_numerical.append(curr_num_state)
        
    return tl_states_numerical


