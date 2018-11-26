class State(object):
    """Structure for State"""
    
    def __init__(self, graph, car_pos):
        self.g = graph
        self.car_pos = car_pos

    def set_car_pos(self, new_car_pos):
        self.g[car_pos]['car_pos'] = False
        self.g[new_car_pos]['car_pos'] = True
        self.car_pos = new_car_pos

    def set_traf_light(self, node, colour):
        self.g[node]['traf_light'] = colour 
        