import random
import logging
import networkx as nx

# Utility values.
GOAL = 100
VIOL = -3
STEP = -1
SPEED = -1


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
        self.visited = [self.cur_pos]
        self.speed_prob = speed_prob

    def act(self, env):

        if self.cur_pos == self.goal:
            logging.debug("Car %d: Already on its goal." % self.id)
            return self.goal
        else:
            g = env.graph
            next_node = self.cur_pos
            higher_utility = self.calculate_utility(next_node, g)
            logging.debug("Car %d eval %d, utility: %f" % (self.id, next_node,
                higher_utility))
            # Get neighbour nodes.
            neighbours = g.neighbors(self.cur_pos)
            # Calculate the best utility.
            for neig_node in neighbours:
                # Go through all neighbour nodes.
                node_util = self.calculate_utility(neig_node, g)
                logging.debug("Car %d eval %d, utility: %f" % (self.id,
                    neig_node, node_util))
                if node_util > higher_utility:
                    higher_utility = node_util
                    next_node = neig_node
            logging.debug("Car %d, chosen node: %d, utility: %f" % (self.id,
                next_node, higher_utility))
            self.modify_speed(env, g, next_node)
            return next_node
                
    def calculate_utility(self, neig_node, g):
        utility = 0
        utility -= self.visited.count(neig_node) / float(10)
        if neig_node == self.goal:
            utility += GOAL
        dist = nx.shortest_path_length(g, neig_node, self.goal)
        utility += dist * STEP

        if neig_node != self.cur_pos:
            # Check traffic light.
            if "signal" in g.node[self.cur_pos]:
                if g.node[self.cur_pos]["signal"]:
                    utility += VIOL
        # Check speed limit.
        if "speed" in g.node[neig_node]:
            node_speed = g.node[neig_node]["speed"]
            if self.speed > node_speed:
                speed_rand = random.random()
                if speed_rand <= self.speed_prob:
                    logging.debug("Car %d updated speed from %d to %d " %
                        (self.id, self.speed, node_speed))
                    # self.speed = node_speed
                    utility += SPEED
                else:
                    utility += VIOL
        # Check prohibition.
        if "prohibition" in g.node[neig_node]:
            utility += VIOL

        return utility

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
            path = nx.shortest_path_length(g, self.cur_pos, self.goal)
            next_node = path[1]
            logging.debug("The next node shall be %d" % next_node)
            # Check probability to go to the next node.
            prob = g[self.cur_pos][next_node]['weight']
            rand_prob = random.random()
            logging.debug("Next node ({}) prob: {}".format(next_node, prob))
            logging.debug("Random prob: {}".format(rand_prob))
            
            car_in_node = None
            if "car" not in g.node[next_node]:
                # Ensure that there is a key for car in node.
                g.node[next_node]["car"] = []
            elif g.node[next_node]["car"]:
                car_in_node = g.node[next_node]["car"][0]

            if rand_prob <= prob and (car_in_node == None
             or car_in_node == self.id):
                # If the rand_prob is smaller than prob and there is no 
                # car in the node other than the car that is going to it.
                # Go to next node.
                self.prev_pos = self.cur_pos
                self.cur_pos = next_node
                self.modify_speed(env, g, next_node)

                env.update_car_position(self, self.prev_pos)
                logging.debug("Car %d moved to node %d" % (self.id,
                    next_node))
                
                return next_node
            else:
                # Go to neighbours.
                neighbours = g.neighbors(self.cur_pos)
                for neig in neighbours:
                    car_in_node =  None
                    if neig == next_node:
                        # Car must not try the same node twice.
                        continue
                        
                    new_prob = g[self.cur_pos][neig]['weight']
                    prob += new_prob
                    rand_prob = random.random()
                    logging.debug("Car {} trying to go to {} with prob {} and rand_prob {}".format(
                        self.id, neig, prob, rand_prob))
                    if "car" not in g.node[neig]:
                        g.node[neig]["car"] = []
                    elif g.node[neig]["car"]:
                        car_in_node = g.node[neig]["car"][0]
                    logging.debug("Car in node: {}".format(car_in_node))
                    logging.debug("Condition to go to neighbour: {} and ({} or {})".format(rand_prob <= prob, car_in_node == None, car_in_node == self.id))
                    if rand_prob <= prob and (car_in_node == None or
                        car_in_node == self.id):
                        self.modify_speed(env, g, neig)
                        self.prev_pos = self.cur_pos
                        self.cur_pos = neig
                        env.update_car_position(self)
                        logging.debug("Car %d moved to node %d" % (self.id,
                            neig))
                        return neig
                logging.debug("Can't move cause no node was available.")
                return self.cur_pos

    def modify_speed(self, env, g, next_node):
        if 'speed' in g.node[next_node]:
            # Update to the next node speed if car's is higher.
            node_speed = g.node[next_node]['speed']
            speed_rand = random.random()
            if(speed_rand <= self.speed_prob and
                self.speed > node_speed):
                logging.debug("Car %d updated speed from %d to %d " %
                    (self.id, self.speed, node_speed))
                self.speed = node_speed
        else:
            speed_rand = random.random()
            if speed_rand <= self.speed_prob:
                new_speed = random.randint(1, env.max_speed)
                logging.debug("Car %d updated speed from %d to %d " %
                    (self.id, self.speed, new_speed))
                self.speed = new_speed