import networkx as nx

# Constants.
VIOLATION = 1
N_VIOLATION = 0


class Agent(object):
    """Build a monitor agent."""
    def __init__(self, nodes):
        """
            :param nodes: list containing node numbers.
            :type nodes: list
            :param norms: norms divided by categories
            :type norms: dict
        """        
        self.nodes = nodes
    
    def verify_violation(plan):
        """
            Verify a violation in a certain state.
        """
        cur_state = plan.cur_state
        prev_state = plan.prev_state
        car_pos = cur_state.car_pos
        states = plan.states

        violation = N_VIOLATION

        if prev_state:
            violation = self.check_speed(prev_state, cur_state)
            if violation:
                return violation

        if cur_state.g[car_pos]['prohibition']:
            return VIOLATION

        if prev_state.g[prev_state.car_pos]['traf_light'] == 'red':
            return VIOLATION

        return N_VIOLATION

    def check_speed(self, prev_state, state):
        """
            Verify if the speed by the distance between two states.
        """
        speed_limit = state.g[state.car_pos]['speed_limit']

        if speed_limit != 'free':

            dist = self._distance(prev_state, state)

            if dist > speed_limit:
                return VIOLATION
        
        return N_VIOLATION

    def _distance(self, s1, s2):
        """
            Measure the distance between two nodes.

            :param s1: state containing graph and car position.
            :type s1: tuple
            :param s2: state containing graph and car position.
            :type s2: tuple
        """
        init = s1.car_pos
        goal = s2.car_pos
        graph = s2.g

        return nx.shortest_path_length(graph, init, goal)