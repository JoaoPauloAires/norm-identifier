class Plan(object):
    """Build the plan."""
    def __init__(self, states):
        """
            :param states: list containing the sequence of states
            :type states: list
        """
        self.states = states
        self.index = 0
        self.cur_state = self.states[self.index]
        self.prev_state = None
    
    def step(self):
        self.prev_state = self.states[self.index]
        self.index += 1
        self.cur_state = self.states[self.index]