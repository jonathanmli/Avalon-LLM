from typing import Any, Tuple, Set, Optional
import numpy as np
from .utils import dict_to_set_of_cartesian_products
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
# NOTE: consider using functools.wraps 

'''
This file contains the abstract classes for the components of the search algorithms
'''

class State(ABC):
    '''
    Abstract class for a state. states are unique and immutable
    '''
    notes: dict
    
    def __init__(self, id, notes: Optional[dict] = None):
        '''
        Args:
            id: id of the state, should be unique, usually the name of the state
            notes: any notes about the state
        '''

        self.id = id
        if notes is None:
            notes = dict()
        self.notes = notes

        # ensure that the state is hashable
        hash(self.id)
    
    def copy(self):
        '''
        Returns a copy of the state
        '''
        return State(self.id, self.notes)

    def __repr__(self):
        return f"State({self.id}, {self.notes})"
    
    def __str__(self):
        return f"State({self.id}, {self.notes})"
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return self.id == other.id
    
    def initial_state_randomize(self, rng: np.random.Generator = np.random.default_rng()):
        '''
        Randomizes the initial state automatically during simulation
        '''
        pass
    
END_STATE = State('END_STATE')

class HiddenState(State):
    '''
    Abstract class for a hidden state. Has additional get information set method
    '''
    def get_information_set(self, actor):
        '''
        Returns the information set for the actor
        '''
        return self.id
    
class ForwardTransitor2(ABC):
    '''
    Abstract class for a forward dynamics transitor
    '''
    def __init__(self):
        pass

    def transition(self, state: State, action: dict)->Tuple[State, dict[Any, float], dict]:
        '''
        Transits to the next state given the current state and action

        Args:
            state: current state
            action: joint action taken by the actors

        Returns:
            next_state: next state
            rewards: reward of the transition for each player (NOTE: not actor)
            notes: notes about the transition
        '''
        state, rewards, notes = self._transition(state, action)
        # assert that action.keys() == rewards.keys()
        # print(action.keys(), rewards.keys())
        # assert set(action.keys()) == set(rewards.keys())
        return state, rewards, notes
    
    @abstractmethod
    def _transition(self, state: State, action: dict)->Tuple[State, dict[Any, float], dict]:
        '''
        Transits to the next state given the current state and action

        Args:
            state: current state
            action: joint action taken by the actors

        Returns:
            next_state: next state
            rewards: reward of the transition for each actor
            notes: notes about the transition
        '''
        pass
    
class ActorEnumerator(ABC):
    '''
    Abstract class for an actor enumerator

    Some conventions on actor names:
        -1: environment
        0: player 1, the main player who is trying to maximize reward
        1: player 2, usually the opponent player who is trying to minimize reward
        2+: other adaptive actors
    '''
    def __init__(self):
        pass
    
    def enumerate(self, state: State)->set:
        '''
        Enumerates the actors that may take actions at the state

        Args:
            state: current state

        Returns:
            actors: set of actors that may take actions at the state
        '''
        return self._enumerate(state)

    @abstractmethod
    def _enumerate(self, state: State)->set:
        '''
        Enumerates the actors that may take actions at the state

        Args:
            state: current state

        Returns:
            actors: set of actors that may take actions at the state
        '''
        pass
    
class ActionEnumerator(ABC):
    '''
    Abstract class for an action enumerator

    Some conventions on actor names:
        -1: environment
        0: player 1, the main player who is trying to maximize reward
        1: player 2, usually the opponent player who is trying to minimize reward
        2+: other adaptive actors
    '''

    def __init__(self):
        pass
    
    def enumerate(self, state: State, actor) -> set:
        '''
        Enumerates the possible actions given the current state and actor

        Args:
            state: current state
            actor: actor to enumerate actions for

        Returns:
            actions: list of actions
        '''
        actions = self._enumerate(state, actor)
        # make sure that actions is not empty
        assert actions
        return actions
    
    @abstractmethod
    def _enumerate(self, state: State, actor) -> set:
        '''
        Enumerates the possible actions given the current state and actor

        Args:
            state: current state
            actor: actor to enumerate actions for

        Returns:
            actions: list of actions
        '''
        pass
    
class PolicyPredictor(ABC):
    '''
    Abstract class for a policy predictor
    '''

    def __init__(self):
        pass
    
    def predict(self, state: State, actions, actor=None)-> dict:
        '''
        Predicts the probabilities over actions given the current state

        Args:
            state: current state
            actions: list of actions

        Returns:
            probs: dictionary of probabilities over actions
        '''
        return self._predict(state, actions, actor)
    
    def _predict(self, state: State, actions, actor=None)-> dict:
        '''
        Predicts the probabilities over actions given the current state

        Args:
            state: current state
            actions: list of actions

        Returns:
            probs: dictionary of probabilities over actions
        '''
        # return uniform distribution by default
        probs = dict()
        for action in actions:
            probs[action] = 1/len(actions)
        return probs
    
class ValueHeuristic2(ABC):
    '''
    Abstract class for a heuristic

    TODO: type the return value of evaluate
    '''

    def __init__(self):
        pass
    
    def evaluate(self, state: State) -> tuple[dict[Any, float], dict]:
        '''
        Evaluates the state

        Args:
            state: current state

        Returns:
            values: values of the state for each player
            notes: notes about the state
        '''
        return self._evaluate(state)
    
    @abstractmethod
    def _evaluate(self, state: State) -> tuple[dict[Any, float], dict]:
        '''
        Evaluates the state

        Args:
            state: current state

        Returns:
            values: values of the state for each player
            notes: notes about the state
        '''
        pass

class ActionHeuristicEnumerator(ABC):
    '''
    Abstract class that packages the action enumerator and value heuristic together
    '''
    def predict(self, state: State) -> dict[Any, float]:
        '''
        Predicts the value of each action given the current state

        Args:
            state: current state

        Returns:
            values: values of the actions
        '''
        return self._predict(state)
    
    @abstractmethod
    def _predict(self, state: State) -> dict[Any, float]:
        '''
        Predicts the value of each action given the current state

        Args:
            state: current state

        Returns:
            values: values of the actions
        '''
        pass
    
class InitialInferencer2(ABC):
    '''
    Abstract class for an initial inferencer 2.0, updated to reduce redundancy
    '''

    def __init__(self, sanity_check = False):
        pass
    
    def predict(self, state: State) -> tuple[dict, dict, dict[tuple[tuple[Any, Any],...],Any], dict[tuple[tuple[Any, Any],...],Any], dict]:
        # TODO: finish type hinting for this function
        # TODO: add 'done' to the return value
        # TODO: add sanity check?
        '''
        Conducts initial inference for algorithms like MCTS

        Args:
            state: current state

        Returns:
            policies: dict from actors to dict of action to probability
            next_state_values: dict from next_state to actors to expected value for the actor of the next state
            intermediate_rewards: dict from (joint) actions to actor to intermediate rewards
            transitions: dict from (joint) actions to next states
            notes: dict of state to notes

        None if the state is terminal
        '''
        policies, next_state_values, intermediate_rewards, transitions, notes = self._predict(state)
        # assert that each actor has at least 1 action (i.e. policies[actor] is not empty for each actor)
        if not all(policies.values()):
            raise ValueError('Each actor must have at least 1 possible action')
        # assert intermediate_rewards and transitions have the same keys
        if not set(intermediate_rewards.keys()) == set(transitions.keys()):
            raise ValueError('Intermediate rewards and transitions must have the same keys (joint actions)')
        # TODO: add more checks
        return policies, next_state_values, intermediate_rewards, transitions, notes

    
    @abstractmethod
    def _predict(self, state: State) -> tuple[dict, dict, dict[tuple[tuple[Any, Any],...],Any], dict[tuple[tuple[Any, Any],...],Any], dict]:
        '''
        Conducts initial inference for algorithms like MCTS

        Args:
            state: current state

        Returns:
            policies: dict from actors to dict of action to probability
            next_state_values: dict from next_state to actors to expected value for the actor of the next state
            intermediate_rewards: dict from (joint) actions to actor to intermediate rewards
            transitions: dict from (joint) actions to next states
            notes: dict of state to notes

        None if the state is terminal
        '''
        pass

    @staticmethod
    def single_actor_convert(policy: dict[Any, float], next_state_values: dict[Any, float], intermediate_rewards: dict[Any, float], transitions: dict[Any, State], actor=0) -> tuple[dict, dict, dict[tuple[tuple[Any, Any],...],Any], dict[tuple[tuple[Any, Any],...],Any]]:
        '''
        Converts the return value of predict for a single actor to the return value of predict for multiple actors

        Args:
            policy: dict of action to probability
            next_state_values: dict of next state to expected value
            intermediate_rewards: dict of action (not joint) to intermediate reward
            transitions: dict of action (not joint) to next state
        '''
        return {actor: policy}, {next_state: {actor: value} for next_state, value in next_state_values.items()}, {((actor, action),): {actor: reward} for action, reward in intermediate_rewards.items()}, {((actor, action),): next_state for action, next_state in transitions.items()}

    def sanity_check(self, actors, policies, next_state_values, intermediate_rewards, transitions):
        '''
        Sanity check for the return value of predict
        '''
        # actors should be a set
        assert isinstance(actors, set)
        # policies should be a dict from actors to dict of action to probability
        assert isinstance(policies, dict)
        for actor, probs in policies.items():
            assert isinstance(actor, int)
            assert isinstance(probs, dict)
            for action, prob in probs.items():
                assert isinstance(action, tuple)
                assert isinstance(prob, float)
        # next_state_values should be a dict from next_state to actors to expected value for the actor of the next state
        assert isinstance(next_state_values, dict)
        for next_state, values in next_state_values.items():
            assert isinstance(next_state, State)
            assert isinstance(values, dict)
            for actor, value in values.items():
                assert isinstance(actor, int)
                assert isinstance(value, float)
        # intermediate_rewards should be a dict from (joint) actions to actor to intermediate rewards
        assert isinstance(intermediate_rewards, dict)
        for joint_action, rewards in intermediate_rewards.items():
            assert isinstance(joint_action, tuple)
            assert isinstance(rewards, dict)
            for actor, reward in rewards.items():
                assert isinstance(actor, int)
                assert isinstance(reward, float)
        # transitions should be a dict from (joint) actions to next states
        assert isinstance(transitions, dict)
        for joint_action, next_state in transitions.items():
            assert isinstance(joint_action, tuple)
            assert isinstance(next_state, State)

class PackageInitialInferencer(InitialInferencer2):

    def __init__(self, transitor: ForwardTransitor2, action_enumerator: ActionEnumerator, 
                 action_predictor: PolicyPredictor, actor_enumerator: ActorEnumerator,
                 value_heuristic: ValueHeuristic2):
        super().__init__()
        self.transitor = transitor
        self.action_enumerator = action_enumerator
        self.action_predictor = action_predictor
        self.actor_enumerator = actor_enumerator
        self.value_heuristic = value_heuristic

    def _predict(self, state: State) -> tuple[dict, dict, dict[tuple[tuple[Any, Any],...],Any], dict[tuple[tuple[Any, Any],...],Any], dict[State,dict]]:
        # predict actors using actor_enumerator
        actors = self.actor_enumerator.enumerate(state)
        # predict actions using action_enumerator for each actor
        actor_to_actions = {actor: self.action_enumerator.enumerate(state, actor) for actor in actors}
        # predict probs using action_predictor for each actor
        actor_to_action_to_probs = {actor: self.action_predictor.predict(state, actor_to_actions[actor], actor) for actor in actors}
        # get joint actions from actor_to_actions. joint actions should be tuples of tuples (actor, action), i.e. joint_action1 = ((actor1, action1), (actor2, action2))
        # joint actions should contain cartesian product of actions for each actor
        joint_actions = dict_to_set_of_cartesian_products(actor_to_actions)

        notes = dict()

        if (actors is None) or (not actors):
            joint_action_to_next_state = dict()
            next_state_to_value = dict()
            joint_action_to_rewards = dict()
            next_state_to_notes = dict()
        else:
            # get transitioned states from transitor for each joint action
            joint_action_to_next_state_rewards_notes = {joint_action: self.transitor.transition(state, {actor: action for actor, action in joint_action}) for joint_action in joint_actions}
            joint_action_to_next_state = {joint_action: joint_action_to_next_state_rewards_notes[joint_action][0] for joint_action in joint_actions}
            joint_action_to_rewards = {joint_action: joint_action_to_next_state_rewards_notes[joint_action][1] for joint_action in joint_actions}

            # get value of each next state using value_heuristic
            next_state_to_value_notes = {next_state: self.value_heuristic.evaluate(next_state) for next_state in joint_action_to_next_state.values()}
            next_state_to_value = {next_state: next_state_to_value_notes[next_state][0] for next_state in joint_action_to_next_state.values()}
            next_state_to_notes = {next_state: next_state_to_value_notes[next_state][1] for next_state in joint_action_to_next_state.values()}
        
        notes['next_state_to_heuristic_notes'] = next_state_to_notes
        return actor_to_action_to_probs, next_state_to_value, joint_action_to_rewards, joint_action_to_next_state, notes
    
class Search(ABC):
    '''
    Abstract class for search algorithms

    The design philosophy is that search algorithms themselves do not contain any data 
    Instead, all data is stored in the graph (see beliefs.py)

    Keeps a record of the total nodes expanded and the nodes expanded in the current search
    '''
    def __init__(self):
        # for recording stats
        self.total_nodes_expanded = 0
        self.nodes_expanded = 0
        
        # logger name should be class name
        self.logger = logging.getLogger(self.__class__.__name__)

    def expand(self, datastructure, state: State):
        '''
        Expand starting from a node
        '''
        return self._expand(datastructure, state)
    
    @abstractmethod
    def _expand(self, datastructure, state: State):
        '''
        Expand starting from a node

        TODO: does it matter which actor is expanding the node?
        '''
        pass
    
    def increment_nodes_expanded(self):
        self.nodes_expanded += 1
        self.total_nodes_expanded += 1
    
    def get_total_nodes_expanded(self):
        return self.total_nodes_expanded
    
    def reset_total_nodes_expanded(self):
        self.total_nodes_expanded = 0

    def get_nodes_expanded(self):
        return self.nodes_expanded
    
    def reset(self):
        self.reset_total_nodes_expanded()

    def get_best_action(self, graph, state: State, actor=0):
        '''
        Returns the best action to take at the state given the search algorithm and graph for the actor (usually the main player)
        '''
        raise NotImplementedError

class Agent(ABC):
    '''
    Abstract class for an agent
    '''
    def __init__(self, player = -1):
        self.player = player

    def set_player(self, player):
        self.player = player

    def act(self, state: State, actions: set,):
        '''
        Chooses an action given the current state and actor

        Args:
            state: current state
            actions: set of actions

        Returns:
            action: chosen action
        '''
        action = self._act(state, actions)
        if action not in actions:
            # print('state', state)
            # print('agent class', self.__class__)
            raise ValueError(f'Action {action} not in actions {actions}')
        return action
    
    @abstractmethod
    def _act(self, state: State, actions: set,):
        '''
        Chooses an action given the current state and actor

        Args:
            state: current state
            actions: set of actions

        Returns:
            action: chosen action
        '''
        raise NotImplementedError
    
class RandomAgent(Agent):

    def __init__(self, rng = np.random.default_rng()):
        super().__init__()
        self.rng = rng

    def _act(self, state: State, actions: Set):
        # return a random action uniformly
        return self.rng.choice(list(actions))
    
class Logger():
    '''
    Abstract class for a logger

    # NOTE: deprecated
    '''
    logs: dict

    def __init__(self, logs: Optional[dict] = None):
        if logs is None:
            logs = dict()
        self.logs = logs
    
    def log(self, name, value):
        '''
        Logs a value
        '''
        self.logs[name] = value

    def get_logs(self):
        '''
        Returns the logs
        '''
        return self.logs
    
    def get_log(self, name):
        '''
        Returns a specific log
        '''
        return self.logs[name]

