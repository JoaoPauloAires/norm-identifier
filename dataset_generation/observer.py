import os
import logging


class Observer(object):
    """Build a monitor agent."""
    def __init__(self, obs_id, nodes, outputfile):
        """
            :param obs_id: Observer ID.
            :type obs_id: int
            :param nodes: list containing node numbers.
            :type nodes: list
            :param norms: norms divided by categories
            :type norms: dict
            :param outputfile: File to save dataset.
            :type outputfile: str
        """
        self.max_node = ''
        self.obs_id = obs_id
        self.nodes = nodes
        open(outputfile, 'w').close() # Reset file.
        self.output = outputfile
        with open(self.output, 'w') as w_file:
            w_file.write("sample class\n")
        # self.max_car_at = ''
        # self.max_traf_light = ''
        # self.max_car_speed = ''
        # self.max_speed_lim = ''
        # self.max_prohibition = ''
        # self.max_enforcer = ''
    
    # def binarize(self, env, node, car_num, car_speed, tf_li, speed_lim,
    #     prohib):
    #     # Calculate current binary representation of predicates.
    #     car_at = bin(car_num * node)[2:]
    #     tf_li = bin(node * tf_li)[2:]
    #     speed_lim = bin(node * speed_lim)[2:]
    #     prohib = bin(node * prohib)[2:]
    #     car_speed = bin(car_num * car_speed)[2:]
    #     node_bin = bin(node)[2:]

    #     # Build a string containing all binary values.
    #     # Add 0's to the begin of the string when needed.
    #     state_bin = ''
    #     # Add car.
    #     state_bin += ((len(self.max_car_at) - len(car_at)) * '0'
    #         ) + car_at
    #     # Add speed.
    #     state_bin += ((len(self.max_car_speed) - len(car_speed))* '0'
    #         ) + car_speed
    #     # Add traffic light.
    #     state_bin += ((len(self.max_traf_light) - len(tf_li)) * '0'
    #         ) + tf_li
    #     # Add speed limit.
    #     state_bin += ((len(self.max_speed_lim) - len(speed_lim)) * '0'
    #         ) + speed_lim
    #     # Add prohibition.
    #     state_bin += ((len(self.max_prohibition) - len(prohib)) * '0'
    #         ) + prohib
    #     # Add node.
    #     state_bin += ((len(self.max_node) - len(node_bin))  * '0'
    #         ) + node_bin
    #     # logging.debug("Built a new binary state: %s" % state_bin)

    #     return state_bin

    # def set_max(self, env):
    #     # Get data.
    #     n_cars = len(env.cars.keys())
    #     n_enforcers = len(env.enfs.keys())
    #     n_nodes = len(env.graph.nodes)
    #     n_tr_li = env.max_tr_li
    #     max_speed = env.max_speed
    #     # Max values for predicates.
    #     car_at = (n_cars + 1)*n_nodes
    #     traf_light = n_nodes*n_tr_li
    #     car_speed = n_cars*max_speed
    #     speed_lim = n_nodes*max_speed
    #     prohib = n_nodes*2  # Considering a binary prohibition.
    #     # Max binarization.
    #     self.max_node = bin(len(env.graph.nodes))[2:]
    #     self.max_car_at = bin(car_at)[2:]
    #     self.max_traf_light = bin(traf_light)[2:]
    #     self.max_car_speed = bin(car_speed)[2:]
    #     self.max_speed_lim = bin(speed_lim)[2:]
    #     self.max_prohibition = bin(prohib)[2:]
    #     self.max_enforcer = bin(n_enforcers)[2:]

    def save_state(self, env, enf_nodes):
        """
            Save current state and if a violation has occurred.
        """
        logging.debug("Starting to process observer %d" % self.obs_id)
        # if not self.max_car_at:
        #     # Set max binary values for further binarization.
        #     self.set_max(env)

        state_bin = ''
        base_bin = ''

        for node in self.nodes:
            car_at = 0
            prohib = 0
            speed_lim = 0
            tf_li = 0
            graph_node = env.graph.node[node]

            # Check car.
            if 'car' in graph_node:
                # if graph_node['car']:
                #     car_id = graph_node['car'][0]
                #     car = env.cars[car_id]
                #     car_num = car_id
                #     car_speed = car.speed
                car_at = 1
            # Check prohibition.
            if 'prohibition' in graph_node:
                # if graph_node['prohibition']:
                prohib = 1
            # Check speed limit.
            if 'speed_limit' in graph_node:
                speed_lim = 1   # graph_node['speed_limit']
            # Check traffic signal.
            if 'signal' in graph_node:
                tf_li = 1    # graph_node['signal']
            # Convert to binary 
            bin_node = '%d%d%d%d' % (car_at, prohib, speed_lim, tf_li)
            # self.binarize(env, node, car_num, car_speed, tf_li, speed_lim, prohib)
            # Add to the state representation.
            base_bin += bin_node
        
        # Get the max number of enforcers and binarize.
        n_enforcers = len(env.enfs.keys())
        max_enforcer = bin(n_enforcers)[2:]

        no_violation = True
        detect_viol = []    # Saves enforcers that already notified violations.
        for node in self.nodes:
            # Run through all nodes in the observer range.
            for enf_node in enf_nodes:
                # Run through nodes monitored by enforcers.
                enf_id, enf_node, violation, _ = enf_node # Get node info.
                if violation:
                    # If the node has a violation notification.
                    if node == enf_node and enf_id not in detect_viol:
                        """
                        Check if the observer observer the node and
                        if the enforcer hasn't notified a violation
                        already.
                        """
                        no_violation = False    # Confirms that a violation happened.
                        detect_viol.append(enf_id)  # Saves the enforcer that notified.
                        enf_id = bin(enf_id)[2:]    # Binarize enforcer's id.
                        tail_bin = ''
                        # Add enforcer.
                        tail_bin += ((len(max_enforcer) - len(enf_id))
                            * '0') + enf_id
                        state_bin = base_bin + tail_bin
                        with open(self.output, 'a') as wrt:
                            logging.debug(
                                "Saving to %s state: %d, violation %d, and %s"
                                % (self.output, node, violation,
                                 state_bin))
                            wrt.write(state_bin + ' ' + str(
                                violation) + '\n')

        logging.debug("No violation status: {}".format(no_violation))
        if no_violation:
            violation = 0 
            tail_bin = len(max_enforcer) * '0'
            state_bin = base_bin + tail_bin
            with open(self.output, 'a') as wrt:
                logging.debug("Saving to %s state: %d, violation %d, and %s"
                    % (self.output, node, violation, state_bin))
                wrt.write(state_bin + ' ' + str(violation) + '\n')