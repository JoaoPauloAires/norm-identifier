import sys
import logging
import argparse
import itertools
import environment
import problem_reader
import networkx as nx
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG, filename='gen_dataset.log',
    filemode='w', format='%(name)s - %(levelname)s - %(message)s')

NODE_PROB = 0.5

# Create a file with all definitions of graph, states, and plan.
class GenDataset(object):
    """Generate Dataset."""
    def __init__(self, env):
        logging.debug("Initiating GenDatsetObj.")
        self.env = env
        
    def save_graph(self):
        nx.draw(self.env.graph)
        plt.savefig("graph.png")

    def run_plans(self):
        # Run over plans.
        prev_encode = ''
        prev_diff = float("inf")
        env_encode = self.encode_env()
        logging.debug("Current state encode: {}".format(env_encode))
        diff = self.check_diff(prev_encode, env_encode)
        logging.debug("Initial difference between states: %d" % diff)
        step = 0

        while diff - prev_diff != 0:
            logging.debug("Starting step %d." % step)

            # Cars.
            for car_id in self.env.cars:
                car = self.env.cars[car_id]
                car.move(self.env)
            # Enforcers.
            enf_nodes = []
            for enf_id in self.env.enfs:
                enf = self.env.enfs[enf_id]
                enf_nodes += enf.verify_violation(self.env)                
            # Observers.
            for obs_id in self.env.obs:
                for enf_node in enf_nodes:
                    obs = self.env.obs[obs_id]
                    enf_id, node, violation, _ = enf_node
                    obs.save_state(self.env, enf_id, node, violation)
            prev_encode = env_encode
            prev_diff = diff
            env_encode = self.encode_env()
            diff = self.check_diff(prev_encode, env_encode)        
    
    def check_diff(self, prev, cur):

        if len(prev) >= len(cur):
            cur += '-' * (len(prev) - len(cur))
            return sum(1 for i in range(len(prev)) if prev[i] != cur[i])

        elif len(cur) > len(prev):
            prev += '-' * (len(cur) - len(prev))
            return sum(1 for i in range(len(cur)) if cur[i] != prev[i])

    def encode_env(self):
        state_conf = ''
        
        for n in self.env.graph.nodes:
            state_conf = state_conf + str(n)

            graph_node = self.env.graph.node[n]

            # Check car.
            if 'car' in graph_node:
                if graph_node['car']:
                    for car_id in graph_node['car']:
                        car = self.env.cars[car_id]
                        speed = car.speed
                        state_conf = state_conf + str(car_id)
                        state_conf = state_conf + str(speed)
            # Check prohibition.
            if 'prohibition' in graph_node:
                if graph_node['prohibition']:
                    state_conf = state_conf + '0'
            # Check speed limit.
            if 'speed' in graph_node:
                state_conf = state_conf + str(graph_node['speed'])

        return state_conf



def main(problem_path):
    # Set graph, plans, and agents by reading from a file.
    G, cars, obs, enfs = problem_reader.read_problem(problem_path)
    env = environment.Environment(G, NODE_PROB, cars, obs, enfs)
    logging.debug("Environment: Nodes: {}".format(env.graph.nodes))
    logging.debug("Environment: Conections: {}".format(env.graph.edges()))
    gd = GenDataset(env)
    # gd = GenDataset(G, plans, agents, output_file)
    gd.run_plans()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate dataset.')
    parser.add_argument('problem_path', type=str,
        help='Path to a file containing problem definitions.')

    args = parser.parse_args()
    main(args.problem_path)