import logging


class Observer(object):
    """Build a monitor agent."""
    def __init__(self, obs_id, nodes, outputfile):
        """
            :param nodes: list containing node numbers.
            :type nodes: list
            :param norms: norms divided by categories
            :type norms: dict
        """
        self.obs_id = obs_id
        self.nodes = nodes
        self.output = outputfile
    
    def save_state(self, env, enf_id, node, violation):
        """
            Save current state and if a violation has occured.
        """
        state_conf = str(enf_id)
        
        if node in self.nodes:
            for n in env.graph.nodes:
                if n in self.nodes:
                    state_conf = state_conf + str(n)

                    graph_node = env.graph.node[n]

                    # Check car.
                    if 'car' in graph_node:
                        if graph_node['car']:
                            for car_id in graph_node['car']:
                                car = env.cars[car_id]
                                speed = car.speed
                                state_conf = state_conf + str(car_id)
                                state_conf = state_conf + str(speed)
                    # Check prohibition.
                    if 'prohibition' in graph_node:
                        if graph_node['prohibition']:
                            state_conf = state_conf + '0'
                    # Check speed limit.
                    if 'speed_limit' in graph_node:
                        state_conf = state_conf + str(
                            graph_node['speed_limit'])

        with open(self.output, 'a') as wrt:
            logging.debug("Saving to %s state: %d and violation %d" % (
                self.output, node, violation))
            wrt.write(state_conf + ' ' + str(violation))