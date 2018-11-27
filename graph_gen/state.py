class State(object):
    """Structure for State"""
    
    def __init__(self, graph, car_pos=None):
        self.g = graph
        self.car_pos = car_pos

    def set_car_pos(self, new_car_pos):
        self.car_pos = new_car_pos

    def set_traf_light(self, node, colour):
        self.g.node[node]['traf_light'] = colour 
    
    def set_prohibition(self, node):
        self.g.node[node]['prohibited'] = True 