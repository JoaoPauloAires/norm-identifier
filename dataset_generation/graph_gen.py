import os
import sys
import logging
import argparse
import itertools
import environment
import problem_reader
import networkx as nx
import matplotlib.pyplot as plt

if not os.path.isdir('./logs'):
    os.mkdir('./logs')
logging.basicConfig(level=logging.DEBUG, filename='logs/gen_dataset.log',
    filemode='w', format='%(name)s - %(levelname)s - %(message)s')

NODE_PROB = 0.5

class GenDataset(object):
    """Generate Dataset."""
    def __init__(self, env, prob_name):
        logging.debug("Initiating GenDatsetObj.")
        self.env = env
        self.prob_name = prob_name
        self.all_obs = self.get_filename()
    
    def get_filename(self):
        if not os.path.isdir('dataset'):
            os.mkdir('./dataset')
        wrt = open("dataset/all_obs_" + self.prob_name + ".txt", 'w')
        wrt.write("sample class\n")
        return wrt

    def save_graph(self):

        values = self.env.graph.nodes()
        options = {
            'node_color': values,
            'node_size': 100,
            # 'width': 3,
        }
        pos = nx.random_layout(self.env.graph)

        nx.draw(self.env.graph, pos, **options)
        if not os.path.isdir('./graphs'):
          os.mkdir('./graphs')
        plt.savefig("graphs/graph_"+self.prob_name+".png")

    def run_plans(self):
        # Run over plans.
        step = 0
        total_cars = len(self.env.cars)
        prev = 100

        while self.env.cars:
            logging.debug("Starting step %d." % step)

            # Cars.
            for car_id in self.env.cars:
                car = self.env.cars[car_id]
                node = car.act(self.env)
                self.env.move_car(car, node)
            # Remove those that are already in their goal nodes.
            self.env.check_cars()
            # Enforcers.
            enf_nodes = []
            for enf_id in self.env.enfs:
                enf = self.env.enfs[enf_id]
                enf_nodes += enf.verify_violation(self.env)
            # Observers.
            for obs_id in self.env.obs:
                obs = self.env.obs[obs_id] # Get current observer.
                obs.save_state(self.env, enf_nodes, self.all_obs)
            self.env.modify()
            step += 1
            cur_prog = len(self.env.cars)
            diff = total_cars - cur_prog
            percent = int((diff/float(total_cars)) * 100)
            if percent % 10 == 0 and percent != prev:
                print "Progress: %d%% of total." % (percent)
                prev = percent
        logging.debug("Finished execution!")
    
def main(problem_path):
    # Set graph, plans, and agents by reading from a file.
    G, cars, obs, enfs, prob_name = problem_reader.read_problem(problem_path)
    env = environment.Environment(G, NODE_PROB, cars, obs, enfs)
    logging.debug("Environment: Nodes: {}".format(env.graph.nodes))
    logging.debug("Environment: Conections: {}".format(env.graph.edges()))
    gd = GenDataset(env, prob_name)
    gd.save_graph()
    gd.run_plans()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate dataset.')
    parser.add_argument('problem_path', type=str,
        help='Path to a file containing problem definitions.')

    args = parser.parse_args()
    main(args.problem_path)
