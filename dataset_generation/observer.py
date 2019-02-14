import os
import random
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

    def save_state(self, env, enf_nodes, all_obs_wrt):
        """
            Save current state and if a violation has occurred.
        """
        logging.debug("Starting to process observer %d" % self.obs_id)

        state_bin = ''
        base_bin = ''

        for node in self.nodes:
            car_at = 0
            high_speed = 0
            prohib = 0
            speed_lim = 0
            tf_li = 0
            graph_node = env.graph.node[node]

            # Check car.
            if 'car' in graph_node:
                if graph_node['car']:
                    car_at = 1
                    if 'speed' in graph_node:
                        node_speed = graph_node['speed']
                        # Check car speed.
                        car_id = graph_node['car'][0]
                        car = env.cars[car_id]
                        car_speed = car.speed
                        if car_speed > node_speed:
                            high_speed = 1
            # Check prohibition.
            if 'prohibition' in graph_node:
                # if graph_node['prohibition']:
                prohib = graph_node['prohibition']
            # Check speed limit.
            if 'speed' in graph_node:
                speed_lim = 1   # graph_node['speed_limit']
            # Check traffic signal.
            if 'signal' in graph_node:
                tf_li = graph_node['signal']
            # Convert to binary 
            bin_node = '%d%d%d%d%d' % (car_at, high_speed, prohib, speed_lim,
                tf_li)
            # self.binarize(env, node, car_num, car_speed, tf_li, speed_lim, prohib)
            # Add to the state representation.
            base_bin += bin_node
        
        # Get the max number of enforcers and binarize.
        n_enforcers = len(env.enfs.keys())
        max_enforcer = bin(n_enforcers)[2:]

        no_violation = True
        enfs = []
        detect_viol = []    # Saves enforcers that already notified violations.
        for node in self.nodes:
            # Run through all nodes in the observer range.
            for enf_node in enf_nodes:
                # Run through nodes monitored by enforcers.
                enf_id, enf_node, violation, _ = enf_node # Get node info.
                if node == enf_node:
                    enfs.append(enf_id)
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
                        # Add enforcer.
                        tail_bin = ((len(max_enforcer) - len(enf_id))
                            * '0') + enf_id
                        state_bin = base_bin + tail_bin
                        with open(self.output, 'a') as wrt:
                            logging.debug(
                                "Saving to %s state: %d, violation %d, and %s"
                                % (self.output, node, violation,
                                 state_bin))
                            all_obs_wrt.write(state_bin + ' ' + str(violation)
                             + '\n')
                            wrt.write(state_bin + ' ' + str(
                                violation) + '\n')

        logging.debug("No violation status: {}".format(no_violation))
        if no_violation:
            violation = 0 
            if enfs:
                enf_id = random.choice(enfs)
                enf_id = bin(enf_id)[2:]
                tail_bin = ((len(max_enforcer) - len(enf_id)) * '0') + enf_id
            else:
                tail_bin = len(max_enforcer) * '0'
            state_bin = base_bin + tail_bin
            with open(self.output, 'a') as wrt:
                logging.debug("Saving to %s state: %d, violation %d, and %s"
                    % (self.output, node, violation, state_bin))
                all_obs_wrt.write(state_bin + ' ' + str(violation) + '\n')
                wrt.write(state_bin + ' ' + str(violation) + '\n')