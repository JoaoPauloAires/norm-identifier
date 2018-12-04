class Observer(object):
    """Build a monitor agent."""
    def __init__(self, nodes, outputfile):
        """
            :param nodes: list containing node numbers.
            :type nodes: list
            :param norms: norms divided by categories
            :type norms: dict
        """        
        self.nodes = nodes
        self.output = outputfile
    
    def save_state(self, env, violation):
        """
            Save current state and if a violation has occured.
        """
        state_conf = ''
        
        for n in env.graph.nodes:
            if n in self.nodes:
                state_conf = state_conf + str(n)

                graph_node = env.graph.node[n]

                # Check car.
                if 'car' in graph_node:
                    if graph_node['car']:
                        for car in graph_node['car']:
                            car_id = car.id
                            speed = car.speed
                            state_conf = state_conf + str(car_id)
                            state_conf = state_conf + str(speed)
                # Check prohibition.
                if 'prohibition' in graph_node:
                    if graph_node['prohibition']:
                        state_conf = state_conf + '0'
                # Check speed limit.
                if 'speed_limit' in graph_node:
                    state_conf = state_conf + str(graph_node['speed_limit'])

        with open(self.output, 'a') as wrt:
            wrt.write(state_conf + ' ' + str(violation))