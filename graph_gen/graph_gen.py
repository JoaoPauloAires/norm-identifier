import networkx as nx

# TODO: Replace prints by logs.

# Rules.
max_vel = 2
forb_state = 5


def distance(s1, s2):
    
    g, init = s1
    _, goal = s2

    return nx.shortest_path_length(g, init, goal)


def verify_plan(plan):

    prev_state = None

    for state in plan:
        graph, pos = state
        if pos == forb_state:
            print "Violation! Entered in a forbidden state."
        if prev_state:
            if distance(prev_state, state) > max_vel:
                print "Violation! Max speed reached!"
                print prev_state[1], state[1]

        prev_state = state


def main():
    nodes = [0,1,2,3,4,5,6,7,8,9]
    arcs = [(0,0), (0,1), (0,7), (1,2), (1,5), (2,3), (3,4), (3,8), (4,9),
            (5,0), (6,3), (7,2), (7,6), (8,3), (8,7), (9,4), (9,8), (9,9)]
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(arcs)
    plan = [(g, 1), (g, 2), (g, 4), (g, 5), (g, 7), (g, 2)]

    verify_plan(plan)


if __name__ == '__main__':
    main()