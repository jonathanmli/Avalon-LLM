import numpy as np
import networkx as nx
# import pygraphviz as pgv
from networkx.drawing.nx_agraph import to_agraph
import matplotlib.pyplot as plt
import time 


# TODO: create tree visualization 
# TODO: refactor the code so that all randomness is determined in random nodes

class Node:
    '''
    Abstract node class for the search algorithms
    '''
    def __init__(self, id, parents=set(), children=set()):
        self.id = id # state of the game that this node represents
        self.parents = parents # parent nodes
        self.children = children # child nodes

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

    def __init__(self, state, parents=set(), children=set()):
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

    def __init__(self, state, parents=set(), children=set(), actions=None, next_states = set()):
        super().__init__(state, parents, children)
        self.value = -np.inf
        self.actions = actions # list of actions
        self.action_to_next_state_probs = dict() # maps action to probabilities over next states (child nodes)
        self.best_action = None # best action to take
        self.next_states = next_states # set of next states (child nodes)

class MinValueNode(ValueNode):
    '''
    State where the opponents are trying to minimize the value by taking actions
    '''

    def __init__(self, state, parents=set(), children=set(), actions=None, next_states = set()):
        super().__init__(state, parents, children)
        self.value = np.inf
        self.actions = actions # actions that the opponent can take
        self.action_to_next_state_probs = dict() # maps action to probabilities over next states
        self.best_action = None # best action to take
        self.next_states = next_states # set of next states (child nodes)

class RandomValueNode(ValueNode):
    '''
    State where the environment progresses to random states
    '''

    def __init__(self, state, parents=set(), children=set(), next_states = None):
        super().__init__(state, parents, children)
        self.value = 0.0
        self.next_states = next_states # set of next states
        self.probs_over_next_states = dict() # maps next state to probability 
        self.best_action = None # best action to take

class SimultaneousValueNode(ValueNode):
    '''
    State where the protagonist and opponents are trying to maximize the value by taking actions simultaneously
    '''

    def __init__(self, state, parents=set(), children=set(), proactions=None, antactions = None, next_states = set()):
        '''
        Args:
            state: state of the game that this node represents
            parents: parent nodes
            children: child nodes
            proactions: actions that the protagonist can take
            antactions: actions that the opponents can take
            next_states: set of next states
        '''
        super().__init__(state, parents, children)
        self.value = -np.inf
        self.proactions = proactions # actions that the protagonist can take
        self.antactions = antactions # actions that the opponents can take
        self.action_to_next_state_probs = dict() # maps action to probabilities over next states (child nodes)
        self.best_action = None # best action to take
        self.next_states = next_states # set of next states (child nodes)

class Graph:
    '''
    A DAG
    '''
    def __init__(self):
        self.id_to_node = dict() # maps id to node

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
    
    def add_state(self, state, parent_states=[], child_states=[]):
        '''
        Adds a state to the tree

        Args:
            state: state to add

        Returns:
            node: node corresponding to the state added
        '''
        parents = set([self.id_to_node[parent_state] for parent_state in parent_states])
        children = set([self.id_to_node[child_state] for child_state in child_states])
        if state not in self.id_to_node:
            # TODO: should 
            # if state.state_type == state.STATE_TYPES[0]:
            if state.state_type == 0:
                node = MaxValueNode(state, parents, children)
            # elif state.state_type == state.STATE_TYPES[1]:
            elif state.state_type == 1:
                node = MinValueNode(state, parents, children)
            # elif state.state_type == state.STATE_TYPES[2]:
            elif state.state_type == 2:
                node = RandomValueNode(state, parents, children)
            else:
                raise NotImplementedError
            self.id_to_node[state] = node
            return node
        else:
            raise ValueError(f"state {state} already exists in the graph")

    
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
    
    def get_best_action(self, state):
        '''
        Returns the best action to take at the state

        Args:
            state: state to get best action of

        Returns:
            best_action: best action to take at the state
        '''
        node = self.id_to_node[state]
        return node.best_action

    def to_networkx(self):
        '''
        Returns the graph as a networkx graph, with values as node.values
        '''
        G = nx.DiGraph()
        for node in self.id_to_node.values():
            # round value to 4 significant figures
            value = round(node.value, 4)
            G.add_node(node.id, value = value)
            for child in node.children:
                G.add_edge(node.id, child.id)
        return G
    
    def to_pygraphviz(self):
        '''
        Returns the graph as a pygraphviz graph, with values as node.values
        '''
        G = to_agraph(self.to_networkx())
        return G
    
    def to_mathplotlib(self):
        '''
        Returns the graph as a matplotlib graph, with values as node.values
        '''
        G = self.to_networkx()

        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_edges(G, pos)
        node_labels = nx.get_node_attributes(G, 'value')
        nx.draw_networkx_labels(G, pos, labels = node_labels)

        # title should be value graph at time 
        title = "Value Graph at time " + str(time.time())
        plt.title(title)
        plt.axis('off')
        return plt