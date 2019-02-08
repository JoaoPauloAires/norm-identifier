import os
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
    # Add Cars.
    logging.debug("Adding cars: {}".format(problem_dict['car']))
    for ind, c in enumerate(problem_dict['car']):
        init_pos, _, _, _ = c[0].split('-')
        init_pos = int(init_pos)
        if 'car' not in G.node[init_pos]:
            G.node[init_pos]['car'] = [ind+1]
        else:
            G.node[init_pos]['car'].append(ind+1)

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
        cars[ind+1] = car.Car(ind+1, int(init_pos), int(goal_pos), int(speed),
            float(speed_prob))

    return cars


def build_enfs(problem_dict):
    enfs = dict()
    logging.debug("Building enfs: {}".format(problem_dict['enf']))
    for ind, enf in enumerate(problem_dict['enf']):
        nodes = map(int, enf[0].split('-'))
        enfs[ind+1] = enforcer.Enforcer(ind+1, nodes)

    return enfs


def build_obs(problem_dict, problem_base_name):
    obs = dict()
    logging.debug("Building obs: {}".format(problem_dict['ob']))
    if not os.path.isdir('./observers'):
        os.mkdir('./observers')
    for ind, ob in enumerate(problem_dict['ob']):
        nodes = map(int, ob[0].split('-'))
        obs_path = 'observers/'+ str(ind+1) + "_" + problem_base_name + ".txt"
        obs[ind+1] = observer.Observer(ind+1, nodes, obs_path)

    return obs


def read_problem(problem_path):
    """
        Read problem file and create the structure for states and plan.
    
        :param problem_path: Path to a file containing a problem description.
        :type problem_path: str
        :return: A graph structure, plans, and agents.
    """
    logging.debug("Start reading from {}".format(problem_path))
    lines = open(problem_path, 'r').readlines()

    base, ext = os.path.splitext(problem_path)
    if "/" in base:
        _, problem_base_name = base.split("/")
    else:
        problem_base_name = base

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
    obs = build_obs(problem_dict, problem_base_name)
    enfs = build_enfs(problem_dict)

    return G, cars, obs, enfs, problem_base_name
