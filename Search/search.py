from beliefs import Graph, MaxValueNode, MinValueNode, RandomValueNode, ValueGraph
from headers import *
from collections import deque

class Search:
    '''
    Abstract class for search algorithms
    '''
    def __init__(self, forward_predictor: ForwardPredictor, forward_enumerator: ForwardEnumerator, 
                 value_heuristic: ValueHeuristic, action_enumerator: ActionEnumerator, 
                 random_state_enumerator: RandomStateEnumerator, random_state_predictor: RandomStatePredictor,
                 opponent_action_enumerator: OpponentActionEnumerator, opponent_action_predictor: OpponentActionPredictor):
        # self.graph = graph
        self.forward_predictor = forward_predictor
        self.forward_enumerator = forward_enumerator
        self.value_heuristic = value_heuristic
        self.action_enumerator = action_enumerator
        self.random_state_enumerator = random_state_enumerator
        self.random_state_predictor = random_state_predictor
        self.opponent_action_enumerator = opponent_action_enumerator
        self.opponent_action_predictor = opponent_action_predictor

    def expand(self, node_id):
        '''
        Expand starting from a node
        '''
        raise NotImplementedError
    
class ValueBFS(Search):
    '''
    Used to perform breadth-first search
    '''
    def __init__(self, forward_predictor: ForwardPredictor, forward_enumerator: ForwardEnumerator, 
                 value_heuristic: ValueHeuristic, action_enumerator: ActionEnumerator, 
                 random_state_enumerator: RandomStateEnumerator, random_state_predictor: RandomStatePredictor,
                 opponent_action_enumerator: OpponentActionEnumerator, opponent_action_predictor: OpponentActionPredictor):
        super().__init__(forward_predictor, forward_enumerator, value_heuristic, action_enumerator, 
                         random_state_enumerator, random_state_predictor,
                         opponent_action_enumerator, opponent_action_predictor)

    def expand(self, graph: ValueGraph, state: State, prev_node = None, depth=3):
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
            
            if state.state_type == state.STATE_TYPES[0]: # max
                max_value = float('-inf')
                max_action = None
                action_to_expected_value = dict()
                

                if node.actions is None:
                    node.actions = self.action_enumerator.enumerate(state)
                
                if not node.next_states:
                    for action in node.actions:
                        node.next_states.update(self.forward_enumerator.enumerate(state, action))
                    
                for next_state in node.next_states:
                    value = self.expand(graph, next_state, node, depth-1)
                    next_state_to_values[next_state] = value

                for action in node.actions:
                    if action not in self.action_to_next_state_probs:
                        self.action_to_next_state_probs[action] = self.forward_predictor.predict(state, action, next_state)
                
                    # calculate expected value
                    expected_value = 0.0
                    for next_state in node.next_states:
                        expected_value += next_state_to_values[next_state] * self.action_to_next_state_probs[action][next_state]
                    action_to_expected_value[action] = expected_value

                    # find max action and max value
                    if expected_value > max_value:
                        max_value = expected_value
                        max_action = action

                node.best_action = max_action
                node.value = max_value
                value = max_value
            
            elif state.state_type == state.STATE_TYPES[1]: # min

                min_value = float('inf')
                min_action = None
                action_to_expected_value = dict()

                if node.actions is None:
                    node.actions = self.opponent_action_enumerator.enumerate(state)

                if not node.next_states:
                    for action in node.actions:
                        node.next_states.add(self.forward_enumerator.enumerate(state, action))

                for next_state in node.next_states:
                    value = self.expand(graph, next_state, node, depth-1)
                    next_state_to_values[next_state] = value

                for action in node.actions:
                    if action not in self.action_to_next_state_probs:
                        self.action_to_next_state_probs[action] = self.forward_predictor.predict(state, action, next_state)
                
                    # calculate expected value
                    expected_value = 0.0
                    for next_state in node.next_states:
                        expected_value += next_state_to_values[next_state] * self.action_to_next_state_probs[action][next_state]
                    action_to_expected_value[action] = expected_value

                    # find min action and min value
                    if expected_value < min_value:
                        min_value = expected_value
                        min_action = action

                node.best_action = min_action
                node.value = min_value
                value = min_value

            elif state.state_type == state.STATE_TYPES[2]: # random 
                value = 0.0
                if state.next_states is None:
                    state.next_states = self.random_state_enumerator.enumerate(state)
                if not state.probs_over_next_states:
                    # Dictionary is empty
                    state.probs_over_next_states = self.random_state_predictor.predict(state, state.next_states)
                for next_state in state.next_states:
                    value += self.expand(graph, next_state, node, depth-1) * state.probs_over_next_states[next_state]
            
            return value

 