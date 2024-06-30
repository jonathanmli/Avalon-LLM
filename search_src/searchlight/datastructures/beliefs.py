import numpy as np
from collections import defaultdict
from typing import Any, Tuple, Set, Optional, Hashable
from ..utils import AbstractLogged


'''
This code contains data structures for storing the search tree
'''

# TODO: refactor the code so that all randomness is determined in random nodes

class Node:
    '''
    Abstract node class for the search algorithms
    '''
    def __init__(self, id, parents=None, children=None, virtual=False):
        self.id = id # usually the state of the game that this node represents
        if parents is None:
            parents = set()
        self.parents = parents # parent nodes
        if children is None:
            children = set()
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
    action_to_next_state: dict
    actor: Optional[int]
    actor_to_value_estimates: dict[int, list]
    action_to_prob_weights: dict
    action_to_actor_to_reward: dict
    is_expanded: bool
    visits: int
    action_to_visits: dict
    actions: set[Hashable]
    notes: dict

    def __init__(self, state, parents=None, children=None, virtual=False):
        super().__init__(state, parents, children, virtual)
        self.state = state # state of the game that this node represents
        self.actor = None # the actor for this state. None if the node is terminal

        self.action_to_next_state = dict() # maps action to next state
        self.action_to_actor_to_reward = dict() # maps action to actor to intermediate reward
        self.action_to_prob_weights = defaultdict(lambda: 1.0) # maps action to probability for the actor

        self.actor_to_value_estimates = defaultdict(list) # maps actor to list of value estimates
        
        self.action_to_visits = defaultdict(int) # maps action to number of visits

        self.visits = 0
        self.notes = dict() # notes about the node

        self.acting_player_information_set = state # the information set for the acting player

        # self.is_expanded = False
        self.actions = set()

    def reset(self):
        self.visits = 0
        self.action_to_visits = defaultdict(int)
        self.actor = None
        self.action_to_next_state = dict()
        self.action_to_actor_to_reward = dict()
        self.action_to_prob_weights = defaultdict(lambda: 1.0)
        self.actor_to_value_estimates = defaultdict(list)
        self.notes = dict()
        
    def get_action_to_probs(self) -> dict:
        '''
        Normalizes the action to prob weights and returns it
        '''
        total = sum(self.action_to_prob_weights.values())
        return {action: prob/total for action, prob in self.action_to_prob_weights.items()}
    
    def get_next_states(self) -> set:
        return set(self.action_to_next_state.values())
    
    def get_actor(self) -> int:
        if self.actor is None:
            raise ValueError("Node is terminal")
        return self.actor
    
    def set_actor(self, actor: int):
        self.actor = actor
    
    def get_actions(self) -> set:
        # return set(self.action_to_next_state.keys())
        return self.actions

    def is_done(self) -> bool:
        return self.actor is None
    
    def get_acting_player_information_set(self) -> Hashable:
        return self.acting_player_information_set

    def set_acting_player_information_set(self, information_set: Hashable):
        self.acting_player_information_set = information_set

    def get_unvisited_actions(self) -> set[Hashable]:
        return self.get_actions() - set(self.action_to_visits.keys())

class InformationSetNode(Node):
    '''
    We need to record here
    - the information set id
    - the nodes in this information set
    - the acting player
    '''

    states_in_set: set
    acting_player: int
    information_set: Hashable
    actions: set

    def __init__(self, information_set: Hashable, acting_player: int, parents=None, children=None, virtual=False):
        super().__init__(information_set, parents, children, virtual)
        self.acting_player = acting_player
        self.states_in_set = set()
        self.information_set = information_set
        self.actions = set()
    
    def add_state(self, state: Hashable):
        self.states_in_set.add(state)
    
    def get_states_in_set(self) -> set:
        return self.states_in_set
    
    def add_actions(self, actions: set):
        self.actions.update(actions)

    def get_actor(self) -> int:
        return self.acting_player
    

class Graph(AbstractLogged):
    '''
    A DAG
    '''
    def __init__(self):
        self.id_to_node = dict() # maps id to node
        super().__init__()

    def get_node(self, id)-> Optional[Any]:
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
    
