import sys
import plan
import copy
import agent
import state
import logging
import networkx as nx


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
    
        :param problem_path: Path to a file containing a problem description.
        :type problem_path: str
        :param b_plans: Flag variable to read plans from problem
            description.
        :type b_plans: bool
        :param b_agents: Flag variable to read agents from problem
            description.
        :type b_agents: bool
        :return: A graph structure, plans, and agents.
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