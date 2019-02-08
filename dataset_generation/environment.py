import random
import logging
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
    def __init__(self, G, node_prob, cars, obs, enfs, max_tr_li=2,
        max_speed=10):
        self.graph = G
        self.node_prob = node_prob
        self.cars = cars
        self.obs = obs
        self.enfs = enfs
        self.max_tr_li = max_tr_li
        self.max_speed = max_speed
        
    def modify(self):
        """
            Check all nodes and modify them using a probability.
        """        
        for n in self.graph.nodes:

            if 'signal' in self.graph.node[n]:
                # If node has a signal, try to change it to red or
                # green.
                if random.random() <= self.node_prob:
                    cur_signal = self.graph.node[n]['signal']
                    self.graph.node[n]['signal'] = (cur_signal * (-1)) + 1

            elif 'prohibition' in self.graph.node[n]:
                # If node has a prohibition status, try to modify it.
                if random.random() <= self.node_prob:
                    cur_status = self.graph.node[n]['prohibition']
                    self.graph.node[n]['prohibition'] = (cur_status * 
                        (-1)) + 1                    

    def update_car_position(self, car, prev_pos):
        """
            Change the current position for car.
        """
        index = self.graph.node[prev_pos]['car'].index(car.id)
        self.graph.node[prev_pos]['car'].pop(index)
        if 'car' in self.graph.node[car.cur_pos]:
            self.graph.node[car.cur_pos]['car'].append(car.id)
        else:
            self.graph.node[car.cur_pos]['car'] = [car.id]

    def check_cars(self):
        """
            Remove cars from environment if they are in the goal node.
        """
        remove_list = []

        for car_id in self.cars:
            car = self.cars[car_id]
            if car.cur_pos == car.goal:
                logging.debug(
                    "Removing car %d from environment. It reached goal %d" % (
                        car_id, car.goal))
                index = self.graph.node[car.cur_pos]['car'].index(car_id)
                self.graph.node[car.cur_pos]['car'].pop(index)

                if car_id in self.cars:
                    remove_list.append(car_id)
                    
        for car_id in remove_list:
            del self.cars[car_id]