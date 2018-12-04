import logging
import networkx as nx

# Constants.
VIOLATION = 1
N_VIOLATION = 0


class Enforcer(object):
    """Build an enforcer agent."""
    def __init__(self, enf_id, nodes):
        """
            :param nodes: list containing node numbers.
            :type nodes: list.
        """
        self.enf_id = enf_id
        self.nodes = nodes
    
    def verify_violation(env):
        """
            Verify a violation in a certain state.
        """
        verification = [] 
        logging.debug("Enforcer %d started the violation verification." % 
            self.enf_id)
        g = env.graph
        for n in g.nodes:
            
            if n in self.nodes:
                violation = N_VIOLATION

                # Check which cars are in this node.
                car_ids = g.node[n]['cars']

                for car_id in car_ids:

                    # Check if the car is in a forbidden state.
                    if 'prohibition' in g.node[n]:
                        if g.node[n]['prohibition']:
                            violation = VIOLATION
                            logging.debug("Enforcer %d detected a prohibition\
                             violation in node %d commited by car %d." % (
                                self.enf_id, n, car_id))
                            verification.append((n, violation))
                            continue
                    # Check if the car didn't cross a red signal.                
                    car = env.cars[car_id]
                    prev_node = car.prev_pos
                    if 'signal' in g.node[prv_node]:
                        if g.node[prev_node]['signal']:
                            violation = VIOLATION
                            logging.debug("Enforcer %d detected a signal \
                             violation in node %d commited by car %d." % (
                                self.enf_id, n, car_id))
                            verification.append((n, violation))
                            continue

                    # Check if car is in a lower or equal speed defined
                    # for the node.
                    if 'speed' in g.node[n]:
                        if car.speed > g.node[n]['speed']:
                            violation = VIOLATION
                            logging.debug("Enforcer %d detected a speed \
                             violation in node %d commited by car %d." % (
                                self.enf_id, n, car_id))
                            verification.append((n, violation))
                            continue