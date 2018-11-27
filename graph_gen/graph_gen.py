import sys
import plan
import copy
import agent
import state
import logging
import argparse
import networkx as nx
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG, filename='gen_dataset.log',
    filemode='w', format='%(name)s - %(levelname)s - %(message)s')


# Create a file with all definitions of graph, states, and plan.

class GenDataset(object):
    """Generate Dataset."""
    def __init__(self, graph, plan, output_file, norms={},
        norm_types=None):
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


def build_graph(problem_dict):
    logging.debug("Building graph")
    assert 'n' in problem_dict, "No nodes found! There is a problem in your \
     input file."
    nodes = map(int, problem_dict['n'][0])
    logging.debug("Nodes: {}".format(nodes))
    arcs = [(int(y[0]), int(y[1])) for y in [x.split('-') for x in 
        problem_dict['c'][0]]]
    logging.debug("Arcs: {}".format(arcs))
    # Create graph.
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(arcs)
    # Add speed limits.
    speed_limits = problem_dict['sp'][0]
    logging.debug("Speed limits: {}".format(speed_limits))
    for speed in speed_limits:
        values = map(int, speed.split('-'))
        limit, nodes = values[0], values[1:]
        # Add limit to nodes.
        for node in nodes:
            G.node[node]['speed_limit'] = limit

    return G


def build_plans(G, problem_dict):
    logging.debug("Started building plans.")
    plans = list() # Save plans.

    assert 'pl' in problem_dict, "No plan found! Check your input file or \
     change the parameters in read_problem function."

    logging.debug("Plans: {}".format(problem_dict['pl']))

    for plan_num in problem_dict['pl']:
        # Run over plans.
        states = list() 
        logging.debug("Reading plan {}".format(plan_num))
        for attrs in problem_dict['pl'][plan_num]:
            attrs = attrs.split(',')
            logging.debug("State with attrs: {}".format(attrs))
            for attr in attrs:
                # Run over attributes.
                g = copy.deepcopy(G)    # Copy the base graph.
                logging.debug("New graph nodes: {}".format(g.nodes()))
                st = state.State(g) # Set state for plan.
                attr = attr.split('-')
                key, values = attr[0], attr[1:]
                if key == 'cp':
                    car_pos = int(values[0])
                    st.set_car_pos(car_pos)
                elif key == 'r':
                    for node in values:
                        st.set_traf_light(int(node), 'red')
                elif key == 'p':
                    for node in values:
                        st.set_prohibition(int(node))
            states.append(st)
        p = plan.Plan(states)
        plans.append(p)

    return plans


def build_agents(problem_dict):
    assert 'ag' in problem_dict, "No agents found, check your input file or change the parameter in read_problem."

    agents = list()

    for nodes in problem_dict['ag']:

        ag = agent.Agent(nodes)
        agents.append(ag)        

    return agents


def read_problem(problem_path, b_plans=True, b_agents=True):
    """
        Read problem file and create the structure for states and plan.
    
        :return: A graph structure and a plan.
    """
    logging.debug("Start reading from {}".format(problem_path))
    lines = open(problem_path, 'r').readlines()
    plan = None
    problem_dict = dict()

    for line in lines:
        key, values = line.strip().split(' ')
        logging.debug("Line: {} and {}".format(key, values))
        if key == 'pl':
            if not plan:
                plan = int(values)
            else:
                plan = int(values)
            if key not in problem_dict:
                problem_dict[key] = dict()
                problem_dict[key][plan] = list()
            else:
                problem_dict[key][plan] = list()
        elif key == 's':
            problem_dict['pl'][plan].append(values)
        else:
            if key not in problem_dict:
                problem_dict[key] = [values.split(',')]
            else:
                problem_dict[key].append(values.split(',')) 

    elems = []

    G = build_graph(problem_dict)
    elems.append(G)
    if b_plans:
        plans = build_plans(G, problem_dict)
        elems.append(plans)
    if b_agents:
        agents = build_agents(problem_dict)
        elems.append(agents)

    return elems


def main(problem_path, output_file):
    # Set graph and plans by reading from a file.
    G, plans, agents = read_problem(problem_path)

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