from flow.networks.base import Network
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
import flow.config as config

import os
import numpy as np

ADDITIONAL_NET_PARAMS = {}

# we create a new network class to specify the expected routes
class MusicRowCorridor(Network):
    
    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))
                
        edges_path = os.path.join(os.path.join(config.PROJECT_PATH,'flow/networks/music_row_16th_ave_edges.txt'))

        self.NET_EDGES = list(np.loadtxt(edges_path,dtype=str,comments=None))
    
        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)
    
    
    def specify_routes(self, net_params):
        
        route_dict = dict.fromkeys(self.NET_EDGES)
        
        
        for i in range(len(self.NET_EDGES)):
            edge = self.NET_EDGES[i]
            route_dict[edge] = self.NET_EDGES[i:]      
        
    
        return route_dict