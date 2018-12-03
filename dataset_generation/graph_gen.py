import sys
import logging
import argparse
import problem_reader
import networkx as nx
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG, filename='gen_dataset.log',
    filemode='w', format='%(name)s - %(levelname)s - %(message)s')

# Create a file with all definitions of graph, states, and plan.

class GenDataset(object):
    """Generate Dataset."""
    def __init__(self, graph, plans, agents, output_file):
        logging.debug("Initiating GenDatsetObj.")
        self.graph = graph
        self.plans = plans
        self.agents = agents
        self.out = output_file
        
    def save_graph(self):
        nx.draw(self.graph)
        plt.savefig("graph.png")

    def run_plans(self):

        # Run over plans.
        for plan in self.plans:

            # Run over plan states.
            for i in range(len(plan.states)):
                
                # Run over agents.
                for ag in self.agents: 
                    # Check violations.
                    ag.verify_violation(plan)

                plan.step()


    def save_to_file(self, state, violation):

        g, pos = state

        logging.debug("Saving state in position %d with class %d" % (pos,
            violation))
        x = ''.join(str(x) + str(y) for x, y in map(list, g.edges())) + str(pos)

        with open(self.out, 'a') as a_file:
            a_file.write("%s %d\n" % (x, violation))


def main(problem_path, output_file):
    # Set graph, plans, and agents by reading from a file.
    G, plans, agents = problem_reader.read_problem(problem_path)

    gd = GenDataset(G, plans, agents, output_file)
    # gd = GenDataset(G, plans, agents, output_file)
    # gd.run_plan()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate dataset.')
    parser.add_argument('problem_path', type=str,
        help='Path to a file containing problem definitions.')
    parser.add_argument('output_file', type=str,
        help='Path to output the generated dataset.')

    args = parser.parse_args()
    main(args.problem_path, args.output_file)