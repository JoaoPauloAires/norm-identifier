import random
import logging
import networkx as nx

class Car(object):
    """Build a car that drives on the environment."""
    def __init__(self, car_id, init_pos, goal):
        """
            :param car_id: identifies the car
            :type car_id: int
            :param init_pos: initial car position
            :type init_pos: int
            :param goal: car goal position
            :type goal: int
        """        
        self.id = car_id
        self.cur_pos = init_pos
        self.goal = goal

    def action(self, env):
        """
            Move the car towards the goal.
        """
        if self.cur_pos == self.goal:
            logging.debug("Car %d: Already on its goal." % self.id)
            return self.goal
        else:
            g = env.graph
            # Calculate shortest path.
            path = nx.shortest_path_length(g, self.cur_pos, self.goal)
            next_node = path[0]
            # Check probability to go to the next node.
            prob = g[self.cur_pos][next_node]['prob']
            
            if random.random() <= prob:
                # Go to next node.
                prev_pos = self.cur_po
                self.cur_pos = next_node
                env.update_car_position(self, prev_pos)
                logging.debug("Car %d moved to node %d" % (self.id, next_node))
                return next_node
            else:
                # Go to neighbours.
                neighbours = g.neighbors(self.cur_pos)
                for neig in neighbours:
                    new_prob = g[self.cur_pos][neig]
                    prob += new_prob
                    if random.random() <= prob:
                        prev_pos = self.cur_pos
                        self.cur_pos = neig
                        env.update_car_position(self, prev_pos)
                        logging.debug("Car %d moved to node %d" % (self.id,
                            neig))
                        return neig