import os
import sys
import time
import json
import random
import logging
import datetime
import argparse
import networkx as nx

logging.basicConfig(level=logging.DEBUG, filename='logs/problem_gen.log',
    filemode='w', format='%(name)s - %(levelname)s - %(message)s')


class ProblemGen(object):
    """Generate random problems to generate datasets."""
    
    def __init__(self, max_nodes, config, output_file=None):

        self.config = json.loads(open(config, 'r').read())
        min_nodes = self.config['min_nodes']
        assert max_nodes >= min_nodes, "max_nodes can't be smaller than min_nodes."
        self.n_nodes = random.randint(min_nodes, max_nodes)
        self.max_connections = int(
            self.n_nodes * self.config['max_connections'])
        self.max_prohibitions = int(
            self.n_nodes * self.config['max_prohibitions'])
        self.max_red_signal = int(
            self.n_nodes * self.config['max_red_signals'])
        self.speed_limits = int(self.n_nodes * self.config['speed_limits'])
        self.min_speed = self.config['min_speed']
        self.max_speed = self.config['max_speed']
        self.max_cars = int(self.n_nodes * self.config['max_cars'])
        self.max_obs = int(self.n_nodes * self.config['max_obs'])
        self.max_enfs = int(self.n_nodes * self.config['max_enfs'])
        self.max_range = int(self.n_nodes * self.config['max_range'])
        self.output_file = output_file

    def create_problem(self):
        
        # Create output_file.
        if not self.output_file:
            logging.debug("Creating output_file file.")
            self.output_file = self.get_outfile()

        logging.debug("Output file: %s" % self.output_file)
        w_file = open(self.output_file, 'w')

        # Create nodes line.
        logging.debug("Adding %d nodes." % self.n_nodes)
        line = self.create_nodes()
        w_file.write(line)
        logging.debug("Created nodes: %s" % line)

        # Create connections.
        line, arcs = self.make_connections()
        w_file.write(line)
        logging.debug("Created connections: %s" % line)

        # Create prohibitions.
        line = self.make_restriction('p')
        w_file.write(line)
        logging.debug("Created prohibitions: %s" % line)

        # Create signals.
        line = self.make_restriction('s')
        w_file.write(line)
        logging.debug("Created signals: %s" % line)

        # Create speed.
        line = self.make_speed_limit()
        w_file.write(line)
        logging.debug("Created speed limits: %s" % line)

        # Create cars.
        line = self.make_cars(arcs)
        w_file.write(line)
        logging.debug("Created cars: %s" % line)

        # Create observers.
        line = self.make_agents('o')
        w_file.write(line)
        logging.debug("Created observers: %s" % line)

        # Create enforcers.
        line = self.make_agents('e')
        w_file.write(line)
        logging.debug("Created enforcers: %s" % line)

        w_file.close()

    def get_outfile(self):
        
        # Get current time.
        currentDT = datetime.datetime.now()
        if not os.path.isdir('./problems'):
            os.mkdir('./problems')
        return "problems/problem_" + currentDT.strftime(
            "%d-%m-%Y_%H-%M-%S") + ".txt"

    def create_nodes(self):
        line = "n "
        
        for i in xrange(self.n_nodes + 1):
            # Add nodes to line.
            line += str(i) + ','
        line = line[:-1] + "\n"

        return line

    def make_restriction(self, r_type):

        if r_type == 'p':
            n_restric = random.randint(1, self.max_prohibitions)
            line = "p "
        elif r_type == 's':
            n_restric = random.randint(1, self.max_red_signal)
            line = "r "
        else:
            sys.exit(1)

        used_nodes = []
        
        for i in xrange(n_restric):
            while True:
                node = random.randint(0, self.n_nodes)
                if node not in used_nodes:
                    line += str(node) + '-'
                    used_nodes.append(node)
                    break
        line = line[:-1] + "\n"
        return line

    def make_cars(self, arcs):

        n_cars = random.randint(1, self.max_cars)
        starts = []
        goals = []
        lines = """"""

        G = nx.DiGraph()
        G.add_nodes_from(range(self.n_nodes+1))
        G.add_weighted_edges_from(arcs)

        for car in xrange(n_cars):
            while True:
                init = random.randint(0, self.n_nodes)
                goal = random.randint(0, self.n_nodes)
                if init != goal and init not in starts and goal not in goals:
                    speed = random.randint(self.min_speed, self.max_speed)
                    prob_speed = random.random()
                    lines += "car " + str(init)+'-'+str(goal)+'-'+str(
                        speed)+'-'+"%.1f" % prob_speed+'\n'
                    break
        return lines

    def make_agents(self, a_type):

        if a_type == 'o':
            n_agents = random.randint(1, self.max_obs)
            logging.debug("Making %d observers." % n_agents)
        elif a_type == 'e':
            n_agents = random.randint(1, self.max_enfs)
            logging.debug("Making %d enforcers." % n_agents)
        else:
            sys.exit(0)
        lines = """"""

        for i in xrange(n_agents):
            if a_type == 'o':
                line = "ob "
            elif a_type == 'e':
                line = "enf "
            r_nodes = random.randint(1, self.max_range)
            used_nodes = []

            for j in xrange(r_nodes):
                while True:
                    node = random.randint(0, self.n_nodes)
                    if node not in used_nodes:
                        line += str(node) + "-"
                        used_nodes.append(node)
                        break
            line = line[:-1] + "\n"
            lines += line
            
        return lines

    def make_speed_limit(self):

        n_limits = random.randint(1, self.speed_limits)
        logging.debug("Making %d speed limits." % n_limits)
        remaining = n_limits
        used_nodes = []
        limits = dict()
        line = "sp "
        for i in xrange(n_limits):
            while True:
                node = random.randint(0, self.n_nodes)
                limit = random.randint(self.min_speed, self.max_speed)

                if node not in used_nodes:
                    if limit in limits:
                        limits[limit].append(node)
                    else:
                        limits[limit] = [node]
                    break

        for limit in limits:
            pre_line = str(limit) + '-'
            for node in limits[limit]:
                pre_line += str(node) + '-'
            pre_line = pre_line[:-1] + ','
            line += pre_line
        line = line[:-1] + "\n"

        return line

    def make_connections(self):

        n_connections = random.randint(self.n_nodes, self.max_connections)
        logging.debug("Making %d connections." % n_connections)
        connections = dict()
        line = "c "
        arcs = []

        # Basic connections.
        for i in xrange(self.n_nodes):
            connections[i] = [i + 1]
        connections[self.n_nodes] = [0]

        # Create connections.
        for i in xrange(n_connections):
            while True:                
                frm = random.randint(0, self.n_nodes)
                to = random.randint(0, self.n_nodes)
                if frm not in connections:
                    connections[frm] = [to]
                    break
                elif len(connections[frm]) < 11:
                    if to not in connections[frm]:
                        connections[frm].append(to)
                        break
        
        # For each connection, define a probability.
        for frm in connections:
            n_conn = len(connections[frm]) # Number of connections left to process.
            prob_used = 0       # Probabilities used so far.
            min_prob = 1
            max_prob_val = 10
            max_prob = int(max_prob_val / n_conn)
            
            # Run over the nodes frm connects to.
            for to in connections[frm]:
                if 1/n_conn == 1:
                    # If there is only one connection, it receives all the
                    # remaining probability.
                    prob = max_prob_val - prob_used
                else:
                    # If there are more than one connection remaining.
                    # Define a maximum value to a probability.
                    prob = random.randint(min_prob, max_prob)
                    prob_used += prob
                    n_conn -= 1

                # Build the line.
                to_line = str(frm) + '-' + str(to) + '-' + "%.1f" % (
                    prob/float(10)) + ','
                arcs.append((frm, to, prob/float(10)))
                line += to_line
 
                if prob_used >= 10:
                    break
        # Modify the end of line.
        line = line[:-1] + '\n'

        return line, arcs


def main(max_nodes, config_path, versions):
    
    logging.debug("Initiating process: max_nodes: {}; config_path: {}; versions: {}".format(max_nodes, config_path, versions))    
    
    for i in xrange(versions):
        p_gen = ProblemGen(max_nodes, config_path)
        p_gen.create_problem()
        time.sleep(1)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Generate problem.')
    parser.add_argument('max_nodes', type=int, 
        help='Max nodes a graph may have.')
    parser.add_argument('config_path', type=str,
        help='Path to a file containing problem configurations.')
    parser.add_argument('versions', type=int,
        help='Number of versions you want to create.')

    args = parser.parse_args()
    main(args.max_nodes, args.config_path, args.versions)