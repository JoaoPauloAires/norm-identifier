import random
import logging
import networkx as nx

class Car(object):
    """Build a car that drives on the environment."""
    def __init__(self, car_id, init_pos, goal, speed, speed_prob):
        """
            :param car_id: identifies the car
            :type car_id: int
            :param init_pos: initial car position
            :type init_pos: int
            :param goal: car goal position
            :type goal: int
            :param speed: initial car speed
            :type speed: int
            :param speed_prob: probability of changing car speed to
            the one in the next node.
            :param speed_prob: float
        """        
        self.id = car_id
        self.prev_pos = init_pos
        self.cur_pos = init_pos
        self.goal = goal
        self.speed = speed
        self.speed_prob = speed_prob

    def move(self, env):
        """
            Move car towards the goal.
        """
        if self.cur_pos == self.goal:
            logging.debug("Car %d: Already on its goal." % self.id)
            return self.goal
        else:
            g = env.graph
            logging.debug("Car {}: cur_pos: {}; goal: {}".format(self.id,
                self.cur_pos, self.goal))
            # Calculate shortest path.
            path = nx.shortest_path(g, self.cur_pos, self.goal)
            next_node = path[1]
            logging.debug("The next node shall be %d" % next_node)
            # Check probability to go to the next node.
            prob = g[self.cur_pos][next_node]['weight']
            rand_prob = random.random()
            logging.debug("Next node ({}) prob: {}".format(next_node, prob))
            logging.debug("Random prob: {}".format(rand_prob))
            if rand_prob <= prob:
                # Go to next node.
                self.prev_pos = self.cur_pos
                self.cur_pos = next_node
                if 'speed' in g.node[next_node]:
                    # Update to the next node speed if car's is higher.
                    node_speed = g.node[next_node]['speed']
                    if(random.random() <= self.speed_prob and
                        self.speed_prob > node_speed):
                        logging.debug("Car %d updated speed from %d to %d " %
                            (self.id, self.speed, node_speed))
                        self.speed = node_speed
                env.update_car_position(self, self.prev_pos)
                logging.debug("Car %d moved to node %d" % (self.id,
                    next_node))
                return next_node
            else:
                # Go to neighbours.
                neighbours = g.neighbors(self.cur_pos)
                for neig in neighbours:
                    new_prob = g[self.cur_pos][neig]['weight']
                    prob += new_prob
                    rand_prob = random.random()
                    logging.debug("Car {} trying to go to {} with prob {} and rand_prob {}".format(
                        self.id, neig, prob, rand_prob))
                    if rand_prob <= prob:
                        self.prev_pos = self.cur_pos
                        self.cur_pos = neig
                        env.update_car_position(self, self.prev_pos)
                        logging.debug("Car %d moved to node %d" % (self.id,
                            neig))
                        return neig