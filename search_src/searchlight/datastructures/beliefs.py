import numpy as np
from ..headers import State
from collections import defaultdict
from typing import Any, Tuple, Set, Optional
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

class ValueNode2(Node):
    action_to_next_state: dict
    actor_to_value_estimates: dict
    actor_to_action_to_prob: dict
    action_to_actor_to_reward: dict
    is_expanded: bool
    visits: int
    actor_to_action_visits: dict
    actor_to_best_action: dict
    done: bool
    notes: dict

    def __init__(self, state, parents=None, children=None, virtual=False):
        super().__init__(state, parents, children, virtual)
        self.state = state # state of the game that this node represents
        self.action_to_next_state = dict() # maps action to next state
        self.actor_to_value_estimates = defaultdict(list) # maps actor to list of value estimates
        self.actor_to_action_to_prob = dict() # maps actor to action to probability
        self.action_to_actor_to_reward = dict() # maps action to actor to intermediate reward
        self.done = False # whether the node is terminal or not

        self.is_expanded = False
        self.visits = 0
        self.actor_to_action_visits = defaultdict(lambda: defaultdict(int)) # maps actor to action to number of visits

        self.actor_to_best_action = dict() # maps actor to best action (mainly for full search)
        self.notes = dict() # notes about the node
    
    def get_joint_actions(self) -> set[tuple[tuple[Any, Any]]]:
        return set(self.action_to_next_state.keys())
    
    def get_next_states(self) -> set:
        return set(self.action_to_next_state.values())
    
    def get_actors(self) -> set:
        return set(self.actor_to_action_to_prob.keys())
    
    def get_actions_for_actor(self, actor) -> set:
        return set(self.actor_to_action_to_prob[actor].keys())

    def is_done(self) -> bool:
        return not self.get_actors()

    def check(self):
        '''
        Checks if all the fields are consistent
        '''
        # keys of self.actor_to_value_estimates should be the same as self.actors
        assert set(self.actor_to_value_estimates.keys()) == self.get_actors()
        # keys of self.actor_to_action_to_prob should be the same as self.actors
        assert set(self.actor_to_action_to_prob.keys()) == self.get_actors()
        # keys of self.action_to_actor_to_reward should be the same as self.actions
        assert set(self.action_to_actor_to_reward.keys()) == self.get_joint_actions()
        # keys of self.action_to_next_state should be the same as self.actions
        assert set(self.action_to_next_state.keys()) == self.get_joint_actions()
        # keys of self.actor_to_action_visits should be the same as self.actors
        assert set(self.actor_to_action_visits.keys()) == self.get_actors()
        # keys of self.actor_to_best_action should be the same as self.actors
        assert set(self.actor_to_best_action.keys()) == self.get_actors()


class Graph(AbstractLogged):
    '''
    A DAG
    '''
    def __init__(self):
        self.id_to_node = dict() # maps id to node
        super().__init__()

    def get_node(self, id)-> Optional[None]:
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
    
