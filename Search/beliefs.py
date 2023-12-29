import numpy as np

class Node:
    '''
    Abstract node class for the search algorithms
    '''
    def __init__(self, id, parents = [], children=[]):
        self.id = id # state of the game that this node represents
        self.parents = parents # parent node
        self.children = children # list of children nodes
        # self.visits = 0
        # self.value = 0.0 # value to be updated by the rollout policy
        # # self.untried_actions = state.legal_actions()

    # def select_child(self):
    #     '''
    #     Selects a child node
    #     '''
    #     return self.children[np.argmax([c.value/c.visits + np.sqrt(2*np.log(self.visits)/c.visits) for c in self.children])]

    # def expand(self, action, state):
    #     '''
    #     Expands the node
    #     '''
    #     child = Node(state, self, action)
    #     self.untried_actions.remove(action)
    #     self.children.append(child)
    #     return child

    # def update(self, result):
    #     '''
    #     Updates the node
    #     '''
    #     self.visits += 1
    #     self.value += result
        
    def __repr__(self):
        return f"Node({self.id})"

    def __str__(self):
        return f"Node({self.id})"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __lt__(self, other):
        return self.id < other.id

    def __gt__(self, other):
        return self.id > other.id

class ValueNode(Node):

    def __init__(self, state, parents = [], children=[], actions = []):
        super().__init__(state, parents, children)
        self.state = state # state of the game that this node represents
        self.value = 0.0 # value to be updated by the rollout policy
        self.visits = 0
        
    def backward(self, value):
        '''
        Updates the node
        '''
        self.visits += 1
        self.value += value

class MaxValueNode(ValueNode):
    '''
    State where the protagonist is trying to maximize the value by taking actions
    '''

    def __init__(self, state, parents = [], children=[], actions = []):
        super().__init__(state, parents, children, actions)
        self.value = -np.inf
        self.actions = actions # list of actions
        self.action_to_next_state_probs = dict() # maps action to probabilities over next states
        self.best_action = None # best action to take

    # def backward(self, value):
    #     '''
    #     Updates the node
    #     '''
    #     self.visits += 1
    #     # only update if value is greater than current value
    #     if value > self.value:
    #         self.value = value
    #         for parent in self.parents:
    #             parent.backward(value)

class MinValueNode(ValueNode):
    '''
    State where the opponents are trying to minimize the value by taking actions
    '''

    def __init__(self, state, parents = [], children=[], actions = []):
        super().__init__(state, parents, children, actions)
        self.value = np.inf
        self.actions = actions # actions that the opponent can take
        self.action_to_next_state_probs = dict() # maps action to probabilities over next states

    # def backward(self, value):
    #     '''
    #     Updates the node
    #     '''
    #     self.visits += 1
    #     # only update if value is less than current value
    #     if value < self.value:
    #         self.value = value
    #         for parent in self.parents:
    #             parent.backward(value)

class RandomValueNode(ValueNode):
    '''
    State where the environment progresses to random states
    '''

    def __init__(self, state, parents = [], children=[], actions = []):
        super().__init__(state, parents, children, actions)
        self.value = 0.0
        self.probs_over_next_states = dict() # maps next state to probability 

class Graph:
    '''
    A DAG
    '''
    def __init__(self):
        self.id_to_node = dict() # maps id to node
        pass

    def get_node(self, id):
        '''
        Returns the node corresponding to the id

        Args:
            id: id to get node of

        Returns:
            node: node corresponding to the id, or None if it does not exist
        '''
        if id not in self.id_to_node:
            return None
        else:
            return self.id_to_node[id]

class ValueGraph(Graph):
    '''
    A DAG where each node represents a state and each edge represents an action
    '''

    def __init__(self):
        super().__init__()
        

    def get_value(self, state):
        '''
        Returns the value of the state

        Args:
            state: state to get value of

        Returns:
            value: value of the state
        '''
        return self.id_to_node[state].value
    
    def add_state(self, state, parent_states = [], child_states = [], actions = []):
        '''
        Adds a state to the tree

        Args:
            state: state to add
        '''
        parents = [self.id_to_node[parent_state] for parent_state in parent_states]
        children = [self.id_to_node[child_state] for child_state in child_states]
        node = ValueNode(state, parents=parents, children=children)
        self.id_to_node[state] = node
    
    def backward(self, state, value):
        '''
        Backward updates the values of parent nodes of the state
        Does not work if there are cycles in the graph

        Args:
            state: state to backward
            value: value to backward
        '''
        node = self.id_to_node[state]

        

    def compute_qvalue(self, state, action):
        '''
        Computes the qvalue of the state and action

        Args:
            state: state to compute qvalue of
            action: action to compute qvalue of

        Returns:
            qvalue: qvalue of the state and action
        '''
        node = self.id_to_node[state]
        qvalue = 0.0
        for child in node.children:
            if child.action == action:
                qvalue += child.value
        return qvalue
