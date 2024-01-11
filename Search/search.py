from Search.beliefs import Graph, MaxValueNode, MinValueNode, RandomValueNode, ValueGraph
from Search.headers import *
from collections import deque
import warnings
import itertools

# TODO: implement MinMaxStats
# TODO: implement UCT search


class Search:
    '''
    Abstract class for search algorithms
    '''
    def __init__(self, forward_transistor: ForwardTransitor,
                 value_heuristic: ValueHeuristic, action_enumerator: ActionEnumerator, 
                 random_state_enumerator: RandomStateEnumerator, random_state_predictor: RandomStatePredictor,
                 opponent_action_enumerator: OpponentActionEnumerator, opponent_action_predictor: OpponentActionPredictor, 
                 opponent_enumerator: OpponentEnumerator = OpponentEnumerator()):
        # self.graph = graph
        self.forward_transistor = forward_transistor
        self.value_heuristic = value_heuristic
        self.action_enumerator = action_enumerator
        self.random_state_enumerator = random_state_enumerator
        self.random_state_predictor = random_state_predictor
        self.opponent_action_enumerator = opponent_action_enumerator
        self.opponent_action_predictor = opponent_action_predictor
        self.opponent_enumerator = opponent_enumerator

    def expand(self, node_id):
        '''
        Expand starting from a node
        '''
        raise NotImplementedError
    
class ValueBFS(Search):
    '''
    Used to perform breadth-first search
    '''
    def __init__(self, forward_transistor: ForwardTransitor,
                 value_heuristic: ValueHeuristic, action_enumerator: ActionEnumerator, 
                 random_state_enumerator: RandomStateEnumerator, random_state_predictor: RandomStatePredictor,
                 opponent_action_enumerator: OpponentActionEnumerator, opponent_action_predictor: OpponentActionPredictor):
        super().__init__(forward_transistor, value_heuristic, action_enumerator, 
                         random_state_enumerator, random_state_predictor,
                         opponent_action_enumerator, opponent_action_predictor)

    def expand(self, graph: ValueGraph, state: State, prev_node = None, depth=3, render = False):
        '''
        Expand starting from a node
        
        Args:
            state: state to expand from
            depth: depth to expand to
            revise: whether to revise the graph or not

        Returns:
            value: updated value of the node
        '''

        node = graph.get_node(state)
        if node is None: # node does not exist, create it
            node = graph.add_state(state)   
        if prev_node is not None:
            node.parents.add(prev_node)
            prev_node.children.add(node)

        if depth == 0:
            value = self.value_heuristic.evaluate(state)
            return value
        else:
            value = 0.0
            next_state_to_values = dict()
            
            if state.state_type == 'control':
                raise NotImplementedError
            
            elif state.state_type == 'adversarial':
                raise NotImplementedError

            elif state.state_type == 'stochastic': # random
                value = 0.0
                if node.actions is None:
                    node.actions = self.random_state_enumerator.enumerate(state)
                if not node.action_to_next_state: # Dictionary is empty
                    for action in node.actions:
                        node.action_to_next_state[action] = self.forward_transistor.transition(state, action)
                if not node.probs_over_actions: # Dictionary is empty
                    node.probs_over_actions = self.random_state_predictor.predict(state, node.actions)
                for next_state in set(node.action_to_next_state.values()):
                    next_state_to_values[next_state] = self.expand(graph, next_state, node, depth-1)

                # add expected value over actions 
                for action in node.actions:
                    next_state = node.action_to_next_state[action]
                    prob = node.probs_over_actions[action]
                    value += prob*next_state_to_values[next_state]

            elif state.state_type == 'simultaneous':

                # enumerate opponents
                if node.opponents is None:
                    node.opponents = self.opponent_enumerator.enumerate(state)

                # enumerate actions
                if not node.adactions:
                    for opponent in node.opponents:
                        node.adactions[opponent] = self.opponent_action_enumerator.enumerate(state, opponent)

                # predict probabilities over actions
                if not node.opponent_to_probs_over_actions:
                    for opponent in node.opponents:
                        node.opponent_to_probs_over_actions[opponent] = self.opponent_action_predictor.predict(state, node.adactions[opponent])

                # enumerate proagonist actions
                if node.proactions is None:
                    node.proactions = self.action_enumerator.enumerate(state)
                        
                # enumerate all possible joint actions. first dimension always protagonist. dimensions after that are opponents
                joint_adversarial_actions = list(itertools.product(node.proactions, *node.adactions.values()))

                # first find probabilities over opponent actions for each opponent
                joint_adversarial_actions_to_probs = dict()
            

            if render:
                plt = graph.to_mathplotlib()
                plt.show()
                
            return value

 