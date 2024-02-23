import numpy as np
import networkx as nx
# import pygraphviz as pgv
from networkx.drawing.nx_agraph import to_agraph
import matplotlib.pyplot as plt
import time 
from Search.headers import State
import matplotlib.colors as mcolors




# TODO: create tree visualization 
# TODO: refactor the code so that all randomness is determined in random nodes

class Node:
    '''
    Abstract node class for the search algorithms
    '''
    def __init__(self, id, parents=None, children=None, virtual=False):
        self.id = id # state of the game that this node represents
        self.parents = parents # parent nodes
        self.children = children # child nodes
        self.virtual = virtual # whether the node is virtual or not
        self.attributes = dict() # other attributes of the node

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

    def __init__(self, state, parents=None, children=None, virtual=False):
        super().__init__(state, parents, children, virtual)
        self.state = state # state of the game that this node represents
        self.values_estimates = [] # TODO: deprecated. values from the rollout policy, in temporal order, for the main player
        self.action_to_next_state = dict() # maps action to next state
        self.proaction_to_value = dict() # maps action to value (ie. Q-value) for player 0 (protagonist)
        self.probs_over_actions = dict() # maps action to probability of taking that action
        self.actor_to_value_estimates = dict() # maps actor to list of value estimates

    def get_mean_value(self):
        '''
        Returns the mean value of the node
        '''
        if len(self.values_estimates) == 0:
            return 0.0
        else:
            return np.mean(self.values_estimates)
        
    def get_last_value(self):
        '''
        Returns the last value of the node
        '''
        if len(self.values_estimates) == 0:
            return 0.0
        else:
            return self.values_estimates[-1]
        
    def get_visits(self):
        '''
        Returns the number of visits to the node
        '''
        return len(self.values_estimates)

class SimultaneousValueNode(ValueNode):
    '''
    State where the (possible multiple) actors are trying to maximize the value by taking actions simultaneously
    '''

    def __init__(self, state, parents=None, children=None, actors = None, next_states = None, virtual=False, 
                 actions = None):
        '''
        Args:
            state: state of the game that this node represents
            parents: parent nodes
            children: child nodes
            proactions: actions that the protagonist can take
            antactions: actions that the opponents can take
            next_states: set of next states
        '''
        super().__init__(state, parents, children, virtual)
        self.actors = actors # set of actors
        self.next_states = next_states # set of next states (child nodes)
        # if self.next_states is None:
        #     self.next_states = frozenset()
        self.action_to_value = dict() # maps action to value (ie. Q-value)
        self.actions = actions # set of joint actions
        # if self.actions is None:
        #     self.actions = frozenset()
        self.actor_to_actions = dict() # maps actor to actions
        self.actor_to_action_to_prob = dict() # maps actor to action to probability
        self.action_to_actor_to_reward = dict() # maps action to actor to intermediate reward

class ValueNode2(Node):

    def __init__(self, state, parents=None, children=None, virtual=False):
        super().__init__(state, parents, children, virtual)
        self.state = state # state of the game that this node represents
        self.action_to_next_state = None # maps action to next state
        self.actor_to_value_estimates = None # maps actor to list of value estimates
        self.actor_to_action_to_prob = None # maps actor to action to probability
        self.action_to_actor_to_reward = None # maps action to actor to intermediate reward
        self.actors = None # set of actors
        self.is_expanded = False
        self.actions = None # set of joint actions
        self.visits = 0
        self.actor_to_action_visits = None # maps actor to action to number of visits

class Graph:
    '''
    A DAG
    '''
    def __init__(self):
        self.id_to_node = dict() # maps id to node

    def get_node(self, id)-> Node:
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
    
    def add_state(self, state: State, parent_states=[], child_states=[]):
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
            # TODO: should be generalized, test if works
            node = SimultaneousValueNode(state, parents, children)
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
        node = self.get_node(state)
        # best action should be argmax of qvalues (node.action_to_value)
        best_action = None
        best_value = -np.inf
        for action, value in node.proaction_to_value.items():
            if value > best_value:
                best_action = action
                best_value = value
        return best_action

    def to_networkx(self):
        '''
        Returns the graph as a networkx graph, with values as node.values
        '''
        G = nx.DiGraph()
        for node in self.id_to_node.values():
            value = node.get_mean_value()
            # round value to 4 significant figures
            value = round(value, 4)
            visits = node.get_visits()
            G.add_node(node.id, value = value, visits = visits)
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

        # create graph
        G = self.to_networkx()

        # Extract 'node.visits' values and normalize them
        visits = [G.nodes[node]['visits'] for node in G.nodes()]
        max_visits = max(visits)
        min_visits = min(visits)
        norm_visits = [(visit - min_visits) / (max_visits - min_visits) for visit in visits]

        # Choose a colormap
        cmap = plt.cm.viridis

        # Map normalized visits to colors
        node_colors = [cmap(norm) for norm in norm_visits]
        
        # Draw the graph
        pos = nx.spring_layout(G)
        fig, ax = plt.subplots()
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors)
        
        node_labels = nx.get_node_attributes(G, 'value')
        nx.draw_networkx_labels(G, pos, labels = node_labels)
        # edge_labels = nx.get_edge_attributes(G, 'action')
        # nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)
        

        # Create an Axes for the color bar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_visits, vmax=max_visits))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Node Visits')

        # title should be value graph at time 
        title = "Value Graph at time " + str(time.time())
        plt.title(title)
        plt.axis('off')
        return plt
    
class ValueGraph2(Graph):
    '''
    A DAG where each node represents a state and each edge represents an action

    Updated version of ValueGraph
    '''

    def __init__(self):
        super().__init__()
    
    def add_state(self, state: State, parent_states=[], child_states=[]):
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
            # TODO: should be generalized, test if works
            node = ValueNode2(state, parents, children)
            self.id_to_node[state] = node
            return node
        else:
            raise ValueError(f"state {state} already exists in the graph")

    def to_networkx(self):
        '''
        Returns the graph as a networkx graph, with values as node.values
        '''
        G = nx.DiGraph()
        for node in self.id_to_node.values():
            value = node.get_mean_value()
            # round value to 4 significant figures
            value = round(value, 4)
            visits = node.get_visits()
            G.add_node(node.id, value = value, visits = visits)
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

        # create graph
        G = self.to_networkx()

        # Extract 'node.visits' values and normalize them
        visits = [G.nodes[node]['visits'] for node in G.nodes()]
        max_visits = max(visits)
        min_visits = min(visits)
        norm_visits = [(visit - min_visits) / (max_visits - min_visits) for visit in visits]

        # Choose a colormap
        cmap = plt.cm.viridis

        # Map normalized visits to colors
        node_colors = [cmap(norm) for norm in norm_visits]
        
        # Draw the graph
        pos = nx.spring_layout(G)
        fig, ax = plt.subplots()
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors)
        
        node_labels = nx.get_node_attributes(G, 'value')
        nx.draw_networkx_labels(G, pos, labels = node_labels)
        # edge_labels = nx.get_edge_attributes(G, 'action')
        # nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)
        

        # Create an Axes for the color bar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_visits, vmax=max_visits))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Node Visits')

        # title should be value graph at time 
        title = "Value Graph at time " + str(time.time())
        plt.title(title)
        plt.axis('off')
        return plt