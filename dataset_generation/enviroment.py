import random
import networkx as nx


class Environment(object):
    """
        Class containing the environment definition.
        The environment is a graph where each node has some attributes.
        Most nodes have sensors, which norm enforcers use to detect violations.
        Three different agents act over the environment:
        1. Cars: Drive through the graph nodes.
        2. Norm Enforcers: Cover some sensors in to capture violations.
        3. Observers: Spread over the graph, they perceive when violations
            occur or not. 
    """
    def __init__(self, G, node_prob, cars, obs, enfs):
        self.graph = G
        self.node_prob = node_prob
        self.cars = cars
        self.obs = obs
        self.enfs = enfs

    def modify(self):
        """
            Check all nodes and modify them using a probability.
        """        
        for n in self.graph.nodes:

            if 'signal' in self.graph.node[n]:

                if random.random() <= self.node_prob:
                    cur_signal = self.graph.node[n]['signal']
                    self.graph.node[n]['signal'] = (cur_signal * (-1)) + 1

    def update_car_position(car, prev_pos):
        """
            Change the current position for car.
        """
        index = self.graph.node[prev_pos]['car'].index(car.id)
        self.graph.node[prev_pos]['car'].pop(index)
        self.graph.node[car.cur_pos]['car'].append(car.id)