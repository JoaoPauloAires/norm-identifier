import os
import sys
import logging
import argparser
import networkx as nx

if not os.path.isdir('./logs'):
    os.mkdir('./logs')
logging.basicConfig(level=logging.DEBUG, filename='logs/graphviz_gen.log',
    filemode='w', format='%(name)s - %(levelname)s - %(message)s')


class GraphvizGen(object):
    """Convert the """
    def __init__(self, arg):
        super(GraphvizGen, self).__init__()
        self.arg = arg
        


def main(problem_path):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate graphviz from problem definitions.')
    parser.add_argument('problem_path', type=str,
        help='Path to a file containing problem definitions.')

    args = parser.parse_args()
    main(args.problem_path)