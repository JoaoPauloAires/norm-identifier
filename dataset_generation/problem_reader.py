import sys
import car
import copy
import logging
import observer
import enforcer
import networkx as nx


def build_graph(problem_dict):
    logging.debug("Building graph")
    assert 'n' in problem_dict, "No nodes found! There is a problem in your \
     input file."
    nodes = map(int, problem_dict['n'][0])
    logging.debug("Nodes: {}".format(nodes))
    arcs = [(int(y[0]), int(y[1]), float(y[2])) for y in [x.split('-') for x
     in problem_dict['c'][0]]]
    logging.debug("Arcs: {}".format(arcs))
    # Create graph.
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(arcs)
    # Add speed limits.
    speed_limits = problem_dict['sp'][0]
    logging.debug("Speed limits: {}".format(speed_limits))
    for speed in speed_limits:
        values = map(int, speed.split('-'))
        limit, nodes = values[0], values[1:]
        # Add limit to nodes.
        for n in nodes:
            G.node[n]['speed'] = limit

    # Set forbidden nodes.
    set_feature(G, problem_dict['p'], 'prohibition')

    # Set red signals.
    set_feature(G, problem_dict['r'], 'signal')

    return G


def set_feature(G, dic, feat_name):
    feature = map(int, dic[0][0].split('-'))
    logging.debug("{} nodes: {}".format(feat_name, feature))
    for n in feature:
        G.node[n][feat_name] = 1


def build_cars(problem_dict):
    cars = dict()
    logging.debug("Building cars: {}".format(problem_dict['car']))
    for ind, c in enumerate(problem_dict['car']):
        init_pos, goal_pos, speed, speed_prob = c[0].split('-')
        cars[ind] = car.Car(ind, init_pos, goal_pos, speed, speed_prob)

    return cars


def build_enfs(problem_dict):
    enfs = dict()
    logging.debug("Building enfs: {}".format(problem_dict['enf']))
    for ind, enf in enumerate(problem_dict['enf']):
        nodes = enf[0].split('-')
        enfs[ind] = enforcer.Enforcer(ind, nodes)

    return enfs


def build_obs(problem_dict):
    obs = dict()
    logging.debug("Building obs: {}".format(problem_dict['ob']))
    for ind, ob in enumerate(problem_dict['ob']):
        nodes = ob[0].split('-')
        obs[ind] = observer.Observer(ind, nodes, str(ind)+'.txt')

    return observer


def read_problem(problem_path):
    """
        Read problem file and create the structure for states and plan.
    
        :param problem_path: Path to a file containing a problem description.
        :type problem_path: str
        :return: A graph structure, plans, and agents.
    """
    logging.debug("Start reading from {}".format(problem_path))
    lines = open(problem_path, 'r').readlines()

    problem_dict = dict()

    for line in lines:
        # Divide lines by key and values.
        key, values = line.strip().split(' ')
        logging.debug("Line: {} and {}".format(key, values))
        if key not in problem_dict:
            # Save each key in a dict.
            problem_dict[key] = [values.split(',')]
        else:
            problem_dict[key].append(values.split(','))

    G = build_graph(problem_dict)
    cars = build_cars(problem_dict)
    obs = build_obs(problem_dict)
    enfs = build_enfs(problem_dict)

    return G, cars, obs, enfs