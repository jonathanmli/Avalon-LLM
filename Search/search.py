from Search.beliefs import ValueGraph
from Search.headers import *
from Search.estimators import *
from collections import deque
import warnings
import itertools
from datetime import datetime

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
                 utility_estimator: UtilityEstimator,
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
        self.utility_estimator = utility_estimator

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
                 opponent_action_enumerator: OpponentActionEnumerator, opponent_action_predictor: OpponentActionPredictor, 
                 utility_estimator: UtilityEstimator):
        super().__init__(forward_transistor, value_heuristic, action_enumerator, 
                         random_state_enumerator, random_state_predictor,
                         opponent_action_enumerator, opponent_action_predictor, 
                         utility_estimator, opponent_enumerator = OpponentEnumerator())

    def expand(self, graph: ValueGraph, state: State, prev_node = None, depth=3, render = False, revise = False):
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

        # check if node is terminal
        if state.is_done():
            value = state.get_reward()
            node.values_estimates.append(value)
            return value

        if depth == 0:
            value = self.value_heuristic.evaluate(state)
            node.values_estimates.append(value)
            utility = self.utility_estimator.estimate(node)
            return utility
        else:
            value = 0.0
            next_state_to_values = dict()
            next_depth = depth if node.virtual else depth -1 # skip virtual nodes
            
            if state.state_type == 'control':
                # enumerate actions
                if node.actions is None or revise:
                    node.actions = self.action_enumerator.enumerate(state)

                # find next states
                if not node.next_states or revise:
                    for action in node.actions:
                        next_state = self.forward_transistor.transition(state, action)
                        node.next_states.add(next_state)
                        node.action_to_next_state[action] = next_state

                # expand next states
                for next_state in node.next_states:
                    next_state_to_values[next_state] = self.expand(graph, next_state, node, next_depth)

                # add action to value
                for action in node.actions:
                    next_state = node.action_to_next_state[action]
                    node.action_to_value[action] = next_state_to_values[next_state]

                # value should be max of actions
                value = max(node.action_to_value.values())

            elif state.state_type == 'adversarial':
                # enumerate opponents
                if node.opponents is None or revise:
                    node.opponents = self.opponent_enumerator.enumerate(state)

                # enumerate actions
                if not node.adactions or revise:
                    for opponent in node.opponents:
                        node.adactions[opponent] = self.opponent_action_enumerator.enumerate(state, opponent)

                # predict probabilities over actions
                if not node.opponent_to_probs_over_actions or revise:
                    for opponent in node.opponents:
                        node.opponent_to_probs_over_actions[opponent] = self.opponent_action_predictor.predict(state, node.adactions[opponent], opponent)

                # enumerate joint adversarial actions
                if node.joint_adversarial_actions is None or revise:
                    node.joint_adversarial_actions = list(itertools.product(*node.adactions.values()))

                # find joint adversarial actions to probabilities over actions
                if not node.joint_adversarial_actions_to_probs or revise:
                    for joint_adversarial_action in node.joint_adversarial_actions:
                        node.joint_adversarial_actions_to_probs[joint_adversarial_action] = 1.0
                        for i, opponent in enumerate(node.opponents):
                            action = joint_adversarial_action[i]
                            prob = node.opponent_to_probs_over_actions[opponent][action]
                            node.joint_adversarial_actions_to_probs[joint_adversarial_action] *= prob
                        
                # find next states
                if not node.next_states or revise:
                    for joint_adversarial_action in node.joint_adversarial_actions:
                        next_state = self.forward_transistor.transition(state, joint_adversarial_action)
                        node.next_states.add(next_state)
                        node.joint_adversarial_actions_to_next_states[joint_adversarial_action] = next_state

                # expand next states
                for next_state in node.next_states:
                    next_state_to_values[next_state] = self.expand(graph, next_state, node, next_depth)

                # add expected value over actions
                for joint_adversarial_action in node.joint_adversarial_actions:
                    prob = node.joint_adversarial_actions_to_probs[joint_adversarial_action]
                    next_state = node.joint_adversarial_actions_to_next_states[joint_adversarial_action]
                    value += prob*next_state_to_values[next_state]

            elif state.state_type == 'stochastic': # random
                if node.actions is None or revise:
                    node.actions = self.random_state_enumerator.enumerate(state)
                if not node.action_to_next_state or revise: # Dictionary is empty
                    for action in node.actions:
                        node.action_to_next_state[action] = self.forward_transistor.transition(state, action)
                if not node.probs_over_actions or revise: # Dictionary is empty
                    node.probs_over_actions = self.random_state_predictor.predict(state, node.actions)
                for next_state in set(node.action_to_next_state.values()):
                    next_state_to_values[next_state] = self.expand(graph, next_state, node, next_depth)

                # add expected value over actions 
                for action in node.actions:
                    next_state = node.action_to_next_state[action]
                    prob = node.probs_over_actions[action]
                    value += prob*next_state_to_values[next_state]

            elif state.state_type == 'simultaneous':
                
                # enumerate opponents
                if node.opponents is None or revise:
                    node.opponents = self.opponent_enumerator.enumerate(state)
                    
                # enumerate adactions
                if not node.adactions or revise:
                    for opponent in node.opponents:
                        node.adactions[opponent] = self.opponent_action_enumerator.enumerate(state, opponent)

                # predict probabilities over actions
                if not node.opponent_to_probs_over_actions or revise:
                    for opponent in node.opponents:
                        node.opponent_to_probs_over_actions[opponent] = self.opponent_action_predictor.predict(state, node.adactions[opponent], player=opponent, prob=True)
                
                # enumerate joint adversarial actions. make sure they are tuples
                if node.joint_adversarial_actions is None or revise:
                    node.joint_adversarial_actions = itertools.product(*node.adactions.values())

                # find joint adversarial actions to probabilities over actions
                if not node.joint_adversarial_actions_to_probs or revise:
                    for joint_adversarial_action in node.joint_adversarial_actions:
                        node.joint_adversarial_actions_to_probs[joint_adversarial_action] = 1.0
                        for i, opponent in enumerate(node.opponents):
                            action = joint_adversarial_action[i]
                            prob = node.opponent_to_probs_over_actions[opponent][action]
                            node.joint_adversarial_actions_to_probs[joint_adversarial_action] *= prob

                # enumerate proagonist actions
                if node.proactions is None or revise:
                    node.proactions = self.action_enumerator.enumerate(state)
                        
                # enumerate all possible joint actions. first dimension always protagonist. dimensions after that are opponents
                if node.joint_actions is None or revise:
                    node.joint_actions = itertools.product(node.proactions, *node.adactions.values())

                # find next states
                # TODO: some weird bug here where node.next_states is not empty but node.joint_actions_to_next_states is empty
                # UPDATE: fixed. dicts are mutable so when I was adding to node.joint_actions_to_next_states, I was adding to the same dict
                if not node.next_states or revise:
                    for joint_action in node.joint_actions:
                        next_state = self.forward_transistor.transition(state, joint_action)
                        node.next_states.add(next_state)
                        node.joint_actions_to_next_states[joint_action] = next_state

                # expand next states
                for next_state in node.next_states:
                    next_state_to_values[next_state] = self.expand(graph, next_state, node, next_depth)

                # reset action_to_value to 0
                node.action_to_value = dict()
                for action in node.proactions:
                    node.action_to_value[action] = 0.0

                # add expected value over actions
                for joint_action in node.joint_actions:
                    prob = node.joint_adversarial_actions_to_probs[joint_action[1:]]
                    next_state = node.joint_actions_to_next_states[joint_action]
                    node.action_to_value[joint_action[0]] += prob*next_state_to_values[next_state]

                # value should be max of actions
                value = max(node.action_to_value.values())
                

            if render and not node.virtual:
                plt = graph.to_mathplotlib()
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'Search/output/output_graph_{timestamp}.png'
                plt.savefig(filename)
                plt.close()
                # plt.show()
            
            node.values_estimates.append(value)
            utility = self.utility_estimator.estimate(node)
            return utility

 