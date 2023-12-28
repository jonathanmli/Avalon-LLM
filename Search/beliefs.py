import numpy as np

class Node:
    '''
    Abstract node class for the search algorithms
    '''
    def __init__(self, state, parent=None, action=None):
        self.state = state # state of the game that this node represents
        self.parent = parent # parent node
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0 # value to be updated by the rollout policy
        # self.untried_actions = state.legal_actions()

    def select_child(self):
        '''
        Selects a child node
        '''
        return self.children[np.argmax([c.value/c.visits + np.sqrt(2*np.log(self.visits)/c.visits) for c in self.children])]

    def expand(self, action, state):
        '''
        Expands the node
        '''
        child = Node(state, self, action)
        self.untried_actions.remove(action)
        self.children.append(child)
        return child

    def update(self, result):
        '''
        Updates the node
        '''
        self.visits += 1
        self.value += result
        
    def __repr__(self):
        return f"Node({self.state})"

    def __str__(self):
        return f"Node({self.state})"

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        return self.state == other.state

    def __lt__(self, other):
        return self.state < other.state

    def __gt__(self, other):
        return self.state > other.state



class ValueTree:
    '''
    A tree where each node represents a state and each edge represents an action
    '''

    def __init__(self, root):
        self.root = root
        self.nodes = [root]
        self.edges = []
        self.state_to_node = {root.state: root} # maps state to node

    def get_value(self, state):
        '''
        Returns the value of the state

        Args:
            state: state to get value of

        Returns:
            value: value of the state
        '''
        return self.state_to_node[state].value