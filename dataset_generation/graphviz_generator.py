import os
import sys
import logging
import argparser
import environment
import problem_reader
import networkx as nx

if not os.path.isdir('./logs'):
    os.mkdir('./logs')
logging.basicConfig(level=logging.DEBUG, filename='logs/graphviz_gen.log',
    filemode='w', format='%(name)s - %(levelname)s - %(message)s')

NODE_PROB = 0.5 # Modification probability of a node, from red to green
                # Or from forbidden to free.


class GraphvizGen(object):
    """
    Convert the networkx into Graphviz.
    More than this, we convert each graph state into a graphviz node.
    Thus, each graphviz node is a state and the connections between
    nodes are possible actions in a state.
    """
    def __init__(self, G, cars, obs, enfs, prob_name, env):
        self.G = G
		self.cars = cars
		self.obs = obs
		self.enfs = enfs
		self.prob_name = prob_name
		self.env = env
        

def main(problem_path):
    G, cars, obs, enfs, prob_name = problem_reader.read_problem(problem_path)
    env = environment.Environment(G, NODE_PROB, cars, obs, enfs)
    GraphvizGen(G, cars, obs, enfs, prob_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate graphviz from problem definitions.')
    parser.add_argument('problem_path', type=str,
        help='Path to a file containing problem definitions.')

    args = parser.parse_args()
    main(args.problem_path)