import logging
import networkx as nx
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG, filename='gen_dataset.log',
    filemode='w', format='%(name)s - %(levelname)s - %(message)s')


# Create a file with all definitions of graph, states, and plan.

class GenDataset(object):
    """Generate Dataset."""
    def __init__(self, graph, plan, output_file, norms={},
        norm_types=NORM_TYPES):
        logging.debug("Initiating GenDatsetObj.")
        self.graph = graph
        self.plan = plan
        self.out = output_file
        self.norms = norms
        self.norm_types = norm_types

    def add_norm(self, norm_type, norm):
        """Add a new norm."""
        logging.debug("Adding new norm w/ type: {} and rule: {}".format(norm_type, norm))
        assert norm_type in self.norm_types, "{} must be in Norm types: {}".format(norm_type, self.norm_types)

        if norm_type in self.norm_types:
            if norm_type in self.norms:
                self.norms[norm_type].append(norm)
            else:
                self.norms[norm_type] = [norm]

        logging.debug("New set of norms: {}".format(self.norms))

    def save_graph(self):
        nx.draw(self.graph)
        plt.savefig("graph.png")

    def verify_plan(self):

        prev_state = None

        for state in self.plan:
            # Run over states.
            graph, pos = state
            violation = 0

            for norm_type in self.norms:

                if norm_type == 'speed':
                    if prev_state:
                        violation = self.check_speed(prev_state, state)
                        if violation:
                            break
                elif norm_type == 'prohibition':
                    violation = self.check_prohibition(state)
                    if violation:
                        break

            self.save_to_file(state, violation)
            prev_state = state

    def save_to_file(self, state, violation):

        g, pos = state

        logging.debug("Saving state in position %d with class %d" % (pos,
            violation))
        x = ''.join(str(x) + str(y) for x, y in map(list, g.edges())) + str(pos)

        with open(self.out, 'a') as a_file:
            a_file.write("%s %d\n" % (x, violation))

    def check_speed(self, prev_state, state):
        
        dist = self._distance(prev_state, state)
        print "dist", dist

        for norm in self.norms['speed']:
            print "norm", norm

            if dist > norm:
                return 1
        
        return 0

    def check_prohibition(self, state):
        _, position = state

        if position in self.norms['prohibition']:
            return 1

        return 0

def main():
    nodes = [0,1,2,3,4,5,6,7,8,9]
    arcs = [(0,0), (0,1), (0,7), (1,2), (1,5), (2,3), (3,4), (3,8), (4,9),
            (5,0), (6,3), (7,2), (7,6), (8,3), (8,7), (9,4), (9,8), (9,9)]
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(arcs)
    plan = [(G, 1), (G, 5), (G, 4), (G, 2), (G, 7), (G, 2)]

    # Set plan by reading from a file.
        

    # nx.draw(G)
    # plt.savefig("graph.png")

    gd = GenDataset(G, plan, 'dataset.txt')
    gd.add_norm('speed', MAX_VEL)
    gd.add_norm('prohibition', FORB_STATE)
    gd.verify_plan()

if __name__ == '__main__':
    main()