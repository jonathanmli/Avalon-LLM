from Search.beliefs import ValueGraph
from Search.headers import *
from Search.estimators import *
from collections import deque
import warnings
import itertools
from datetime import datetime
import numpy as np

# TODO: implement MinMaxStats
# TODO: implement UCT search


    


class Search:
    '''
    Abstract class for search algorithms
    '''
    def __init__(self, forward_transistor: ForwardTransitor,
                 value_heuristic: ValueHeuristic, actor_enumerator: ActorEnumerator,
                 action_enumerator: ActionEnumerator, action_predictor: ActionPredictor,
                 utility_estimator: UtilityEstimator):
        # self.graph = graph
        self.forward_transistor = forward_transistor
        self.value_heuristic = value_heuristic
        self.action_enumerator = action_enumerator
        self.utility_estimator = utility_estimator
        self.actor_enumerator = actor_enumerator
        self.action_predictor = action_predictor

        # for recording stats
        self.total_nodes_expanded = 0
        self.nodes_expanded = 0

    def expand(self, node_id):
        '''
        Expand starting from a node
        '''
        return self._expand(node_id)
        
    def _expand(self, node_id):
        '''
        Expand starting from a node
        '''
        raise NotImplementedError
    
    def get_total_nodes_expanded(self):
        return self.total_nodes_expanded
    
    def reset_total_nodes_expanded(self):
        self.total_nodes_expanded = 0

    def get_nodes_expanded(self):
        return self.nodes_expanded

# TODO: refactor this to use the new model headers
# class ValueBFS(Search):
#     '''
#     Used to perform breadth-first search
#     '''
#     def __init__(self, forward_transistor: ForwardTransitor,
#                  value_heuristic: ValueHeuristic, action_enumerator: ActionEnumerator, 
#                  random_state_enumerator: RandomStateEnumerator, random_state_predictor: RandomStatePredictor,
#                  opponent_action_enumerator: OpponentActionEnumerator, opponent_action_predictor: OpponentActionPredictor, 
#                  utility_estimator: UtilityEstimator):
#         super().__init__(forward_transistor, value_heuristic, action_enumerator, 
#                          random_state_enumerator, random_state_predictor,
#                          opponent_action_enumerator, opponent_action_predictor, 
#                          utility_estimator, opponent_enumerator = OpponentEnumerator())
#         self.opponent_action_predictor = opponent_action_predictor

#     def expand(self, graph: ValueGraph, state: State, prev_node = None, depth=3, render = False, revise = False):
#         '''
#         Expand starting from a node
        
#         Args:
#             state: state to expand from
#             depth: depth to expand to
#             revise: whether to revise the graph or not

#         Returns:
#             value: updated value of the node
#         '''

#         node = graph.get_node(state)
#         if node is None: # node does not exist, create it
#             node = graph.add_state(state)   
#         if prev_node is not None:
#             node.parents.add(prev_node)
#             prev_node.children.add(node)

#         # check if node is terminal
#         if state.is_done():
#             value = state.get_reward()
#             node.values_estimates.append(value)
#             return value

#         if depth == 0:
#             value = self.value_heuristic.evaluate(state)
#             node.values_estimates.append(value)
#             utility = self.utility_estimator.estimate(node)
#             return utility
#         else:
#             value = 0.0
#             next_state_to_values = dict()
#             next_depth = depth if node.virtual else depth -1 # skip virtual nodes
            
#             if state.state_type == 'control':
#                 # enumerate actions
#                 if node.actions is None or revise:
#                     node.actions = self.action_enumerator.enumerate(state)

#                 # find next states
#                 if not node.next_states or revise:
#                     for action in node.actions:
#                         next_state = self.forward_transistor.transition(state, action)
#                         node.next_states.add(next_state)
#                         node.action_to_next_state[action] = next_state

#                 # expand next states
#                 for next_state in node.next_states:
#                     next_state_to_values[next_state] = self.expand(graph, next_state, node, next_depth)

#                 # add action to value
#                 for action in node.actions:
#                     next_state = node.action_to_next_state[action]
#                     node.action_to_value[action] = next_state_to_values[next_state]

#                 # value should be max of actions
#                 value = max(node.action_to_value.values())

#             elif state.state_type == 'adversarial':
#                 # enumerate opponents
#                 if node.opponents is None or revise:
#                     node.opponents = self.opponent_enumerator.enumerate(state)

#                 # enumerate actions
#                 if not node.adactions or revise:
#                     for opponent in node.opponents:
#                         node.adactions[opponent] = self.opponent_action_enumerator.enumerate(state, opponent)

#                 # predict probabilities over actions
#                 if not node.opponent_to_probs_over_actions or revise:
#                     for opponent in node.opponents:
#                         node.opponent_to_probs_over_actions[opponent] = self.opponent_action_predictor.predict(state, node.adactions[opponent], opponent)

#                 # enumerate joint adversarial actions
#                 if node.joint_adversarial_actions is None or revise:
#                     node.joint_adversarial_actions = list(itertools.product(*node.adactions.values()))

#                 # find joint adversarial actions to probabilities over actions
#                 if not node.joint_adversarial_actions_to_probs or revise:
#                     for joint_adversarial_action in node.joint_adversarial_actions:
#                         node.joint_adversarial_actions_to_probs[joint_adversarial_action] = 1.0
#                         for i, opponent in enumerate(node.opponents):
#                             action = joint_adversarial_action[i]
#                             prob = node.opponent_to_probs_over_actions[opponent][action]
#                             node.joint_adversarial_actions_to_probs[joint_adversarial_action] *= prob
                        
#                 # find next states
#                 if not node.next_states or revise:
#                     for joint_adversarial_action in node.joint_adversarial_actions:
#                         next_state = self.forward_transistor.transition(state, joint_adversarial_action)
#                         node.next_states.add(next_state)
#                         node.joint_adversarial_actions_to_next_states[joint_adversarial_action] = next_state

#                 # expand next states
#                 for next_state in node.next_states:
#                     next_state_to_values[next_state] = self.expand(graph, next_state, node, next_depth)

#                 # add expected value over actions
#                 for joint_adversarial_action in node.joint_adversarial_actions:
#                     prob = node.joint_adversarial_actions_to_probs[joint_adversarial_action]
#                     next_state = node.joint_adversarial_actions_to_next_states[joint_adversarial_action]
#                     value += prob*next_state_to_values[next_state]

#             elif state.state_type == 'stochastic': # random
#                 if node.actions is None or revise:
#                     node.actions = self.random_state_enumerator.enumerate(state)
#                 if not node.action_to_next_state or revise: # Dictionary is empty
#                     for action in node.actions:
#                         node.action_to_next_state[action] = self.forward_transistor.transition(state, action)
#                 if not node.probs_over_actions or revise: # Dictionary is empty
#                     node.probs_over_actions = self.random_state_predictor.predict(state, node.actions)
#                 for next_state in set(node.action_to_next_state.values()):
#                     next_state_to_values[next_state] = self.expand(graph, next_state, node, next_depth)

#                 # add expected value over actions 
#                 for action in node.actions:
#                     next_state = node.action_to_next_state[action]
#                     prob = node.probs_over_actions[action]
#                     value += prob*next_state_to_values[next_state]

#             elif state.state_type == 'simultaneous':
                
#                 # enumerate opponents
#                 if node.opponents is None or revise:
#                     node.opponents = self.opponent_enumerator.enumerate(state)
                    
#                 # enumerate adactions
#                 if not node.adactions or revise:
#                     for opponent in node.opponents:
#                         node.adactions[opponent] = self.opponent_action_enumerator.enumerate(state, opponent)

#                 # predict probabilities over actions
#                 if not node.opponent_to_probs_over_actions or revise:
#                     for opponent in node.opponents:
#                         node.opponent_to_probs_over_actions[opponent] = self.opponent_action_predictor.predict(state, node.adactions[opponent], player=opponent, prob=True)
                
#                 # enumerate joint adversarial actions. make sure they are tuples
#                 if node.joint_adversarial_actions is None or revise:
#                     node.joint_adversarial_actions = itertools.product(*node.adactions.values())

#                 # find joint adversarial actions to probabilities over actions
#                 if not node.joint_adversarial_actions_to_probs or revise:
#                     for joint_adversarial_action in node.joint_adversarial_actions:
#                         node.joint_adversarial_actions_to_probs[joint_adversarial_action] = 1.0
#                         for i, opponent in enumerate(node.opponents):
#                             action = joint_adversarial_action[i]
#                             prob = node.opponent_to_probs_over_actions[opponent][action]
#                             node.joint_adversarial_actions_to_probs[joint_adversarial_action] *= prob

#                 # enumerate proagonist actions
#                 if node.proactions is None or revise:
#                     node.proactions = self.action_enumerator.enumerate(state)
                        
#                 # enumerate all possible joint actions. first dimension always protagonist. dimensions after that are opponents
#                 if node.joint_actions is None or revise:
#                     node.joint_actions = itertools.product(node.proactions, *node.adactions.values())

#                 # find next states
#                 # TODO: some weird bug here where node.next_states is not empty but node.joint_actions_to_next_states is empty
#                 # UPDATE: fixed. dicts are mutable so when I was adding to node.joint_actions_to_next_states, I was adding to the same dict
#                 if not node.next_states or revise:
#                     for joint_action in node.joint_actions:
#                         next_state = self.forward_transistor.transition(state, joint_action)
#                         node.next_states.add(next_state)
#                         node.joint_actions_to_next_states[joint_action] = next_state

#                 # expand next states
#                 for next_state in node.next_states:
#                     next_state_to_values[next_state] = self.expand(graph, next_state, node, next_depth)

#                 # reset action_to_value to 0
#                 node.action_to_value = dict()
#                 for action in node.proactions:
#                     node.action_to_value[action] = 0.0

#                 # add expected value over actions
#                 for joint_action in node.joint_actions:
#                     prob = node.joint_adversarial_actions_to_probs[joint_action[1:]]
#                     next_state = node.joint_actions_to_next_states[joint_action]
#                     node.action_to_value[joint_action[0]] += prob*next_state_to_values[next_state]

#                 # value should be max of actions
#                 value = max(node.action_to_value.values())
                

#             if render and not node.virtual:
#                 plt = graph.to_mathplotlib()
#                 timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#                 filename = f'Search/output/output_graph_{timestamp}.png'
#                 plt.savefig(filename)
#                 plt.close()
#                 # plt.show()
            
#             node.values_estimates.append(value)
#             utility = self.utility_estimator.estimate(node)
#             return utility

class SMMinimax(Search):
    '''
    Used to perform breadth-first search

    This only works if the opponent is a single agent and the game is a zero-sum game
    '''
    def __init__(self, forward_transistor: ForwardTransitor,
                 value_heuristic: ValueHeuristic, actor_enumerator: ActorEnumerator,
                 action_enumerator: ActionEnumerator, action_predictor: ActionPredictor,
                 utility_estimator: UtilityEstimator):
        
        super().__init__(forward_transistor, value_heuristic, actor_enumerator,
                         action_enumerator, action_predictor, utility_estimator)
        
    def expand(self, graph: ValueGraph, state: State, 
               prev_node = None, depth=3, render = False, 
               revise = False, oracle = True, 
               node_budget=None):
        '''
        Expand starting from a node
        
        Args:
            state: state to expand from
            depth: depth to expand to
            render: whether to render the graph or not
            revise: whether to revise the graph or not
            oracle: whether the opponent always plays the best response or not as if they knew the protagonist's policy

        Returns:
            value: updated value of the node
        '''
        self.nodes_expanded = 0
        return self._expand(graph, state, prev_node, depth, render, revise, oracle, node_budget)

    def _expand(self, graph: ValueGraph, state: State, 
                prev_node = None, depth=3, render = False, 
                revise = False, oracle = True, 
                node_budget=None):
        '''
        Expand starting from a node
        
        Args:
            state: state to expand from
            depth: depth to expand to
            render: whether to render the graph or not
            revise: whether to revise the graph or not
            oracle: whether the opponent always plays the best response or not as if they knew the protagonist's policy

        Returns:
            value: updated value of the node
        '''

        self.total_nodes_expanded += 1
        self.nodes_expanded += 1

        if not oracle:
            raise NotImplementedError
        
        # print('Now exploring state', state)

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
            utility = self.utility_estimator.estimate(node)
            # print('Terminal state', state, 'value', utility)
            return utility

        if depth == 0:
            value = self.value_heuristic.evaluate(state)
            node.values_estimates.append(value)
            utility = self.utility_estimator.estimate(node)
            # print('Depth 0 state', state, 'value', utility)
            return utility
        elif node_budget is not None and self.nodes_expanded > node_budget:
            # use heuristic to estimate value if node budget is exceeded
            value = self.value_heuristic.evaluate(state)
            node.values_estimates.append(value)
            utility = self.utility_estimator.estimate(node)
            # print('Node budget state', state, 'value', utility)
            return utility
        else:
            value = 0.0
            next_state_to_values = dict()
            next_depth = depth if node.virtual else depth -1 # skip virtual nodes

            # enumerate actors
            if node.actors is None or revise:
                node.actors = self.actor_enumerator.enumerate(state)

            # print('node actors', node.actors)

            if -1 in node.actors: # random
                assert len(node.actors) == 1

                if node.actions is None or revise:
                    node.actions = self.action_enumerator.enumerate(state, -1)

                # print('node actions', node.actions)

                if not node.action_to_next_state or revise: # Dictionary is empty
                    for action in node.actions:
                        # print('action', action)
                        node.action_to_next_state[action] = self.forward_transistor.transition(state, {-1: action})
                if not node.probs_over_actions or revise: # Dictionary is empty
                    node.probs_over_actions = self.action_predictor.predict(state, node.actions, -1)
                # print('probs over actions', node.probs_over_actions)
                for next_state in set(node.action_to_next_state.values()):
                    # print('next state', next_state)
                    next_state_to_values[next_state] = self._expand(graph, next_state, node, next_depth,
                                                                    revise=revise, oracle=oracle, node_budget=node_budget)
                
                # print('next state to values', next_state_to_values)

                # add expected value over actions 
                for action in node.actions:
                    next_state = node.action_to_next_state[action]
                    prob = node.probs_over_actions[action]
                    value += prob*next_state_to_values[next_state]

                # print('Random state', state, 'expected value', value)

            elif (0 in node.actors) and (1 in node.actors): # simultaneous
                assert len(node.actors) == 2

                # enumerate actions for each actor
                if not node.actor_to_actions or revise:
                    for actor in node.actors:
                        node.actor_to_actions[actor] = self.action_enumerator.enumerate(state, actor)

                # print('actor to actions', node.actor_to_actions)

                # print([x for x in itertools.product(*node.actor_to_actions.values())])
            
                # enumerate all possible joint actions as tuples of tuples (actor,action) pairs
                if node.actions is None or revise:
                    actors = node.actor_to_actions.keys()
                    node.actions = [tuple(zip(actors, x)) for x in itertools.product(*node.actor_to_actions.values())]
                
                # print('actions', node.actions)

                # find next states
                if node.next_states is None or revise:
                    node.next_states = set()
                    # print('next states not found')
                    for joint_action in node.actions:
                        next_state = self.forward_transistor.transition(state, dict(joint_action))
                        node.next_states.add(next_state)
                        node.action_to_next_state[joint_action] = next_state
                        # print('joint action', joint_action, 'next state', next_state)
                    node.next_states = frozenset(node.next_states)

                # print('next states', node.next_states)

                # expand next states
                for next_state in node.next_states:
                    # print('next state', next_state)
                    # print('next state hash', hash(next_state))
                    next_state_to_values[next_state] = self._expand(graph, next_state, node, next_depth,
                                                                    revise=revise, oracle=oracle, node_budget=node_budget)

                # print('next state to values', next_state_to_values)

                # for each protagonist action, find the best response of the opponent
                opponent_best_responses = dict() # mapping from protagonist action to tuple of (opponent action, value)

                # set br values to infinity
                for proaction in node.actor_to_actions[0]:
                    opponent_best_responses[proaction] = (None, float('inf'))

                # find best response for each protagonist action
                for joint_action in node.actions:
                    proaction = dict(joint_action)[0]
                    value = next_state_to_values[node.action_to_next_state[joint_action]]
                    # print('joint action', joint_action, 'value', value)
                    if value < opponent_best_responses[proaction][1]:
                        opponent_best_responses[proaction] = (joint_action[1], value)

                # print('opponent best responses', opponent_best_responses)

                # set action to value to be opponent best response
                for proaction in node.actor_to_actions[0]:
                    node.proaction_to_value[proaction] = opponent_best_responses[proaction][1]

                # value should be max of actions
                value = max(node.proaction_to_value.values())

                # print('Simultaneous state', state, 'minimax value', value)
            
            else:
                print('node actors', node.actors)
                print('state', state)
                print('state actors', state.actors)
                raise NotImplementedError
                
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
        
class SMAlphaBetaMinimax(Search):
    '''
    Used to perform alpha-beta pruning search

    This only works if the opponent is a single agent and the game is a zero-sum game
    '''
    def __init__(self, forward_transistor: ForwardTransitor,
                 value_heuristic: ValueHeuristic, actor_enumerator: ActorEnumerator,
                 action_enumerator: ActionEnumerator, action_predictor: ActionPredictor,
                 utility_estimator: UtilityEstimator):
        
        super().__init__(forward_transistor, value_heuristic, actor_enumerator,
                         action_enumerator, action_predictor, utility_estimator)
        
    def expand(self, graph: ValueGraph, state: State,
                prev_node = None, depth=3, render = False, 
                revise = False, oracle = True, 
                alpha = -float('inf'), beta = float('inf'), threshold = 0.0,
                node_budget=None):
          '''
          Expand starting from a node
          
          Args:
                state: state to expand from
                depth: depth to expand to
                render: whether to render the graph or not
                revise: whether to revise the graph or not
                oracle: whether the opponent always plays the best response or not as if they knew the protagonist's policy
                alpha: alpha value for alpha-beta pruning
                beta: beta value for alpha-beta pruning
                threshold: threshold for alpha-beta pruning
    
          Returns:
                value: updated value of the node
          '''
          self.nodes_expanded = 0
          return self._expand(graph, state, prev_node, depth, render, revise, oracle, alpha, beta, threshold, node_budget)

    def _expand(self, graph: ValueGraph, state: State, 
               prev_node = None, depth=3, render = False, 
               revise = False, oracle = True, 
               alpha = -float('inf'), beta = float('inf'), threshold = 0.0,
               node_budget=None):
        '''
        Expand starting from a node
        
        Args:
            state: state to expand from
            depth: depth to expand to
            render: whether to render the graph or not
            revise: whether to revise the graph or not
            oracle: whether the opponent always plays the best response or not as if they knew the protagonist's policy
            alpha: alpha value for alpha-beta pruning
            beta: beta value for alpha-beta pruning
            threshold: threshold for alpha-beta pruning

        Returns:
            value: updated value of the node
        '''

        self.total_nodes_expanded += 1
        self.nodes_expanded += 1

        if not oracle:
            raise NotImplementedError
        
        # print('Now exploring state', state)

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
            utility = self.utility_estimator.estimate(node)
            # print('Terminal state', state, 'value', utility)
            return utility

        if depth == 0:
            value = self.value_heuristic.evaluate(state)
            node.values_estimates.append(value)
            utility = self.utility_estimator.estimate(node)
            # print('Depth 0 state', state, 'value', utility)
            return utility
        elif node_budget is not None and self.nodes_expanded > node_budget:
            # use heuristic to estimate value if node budget is exceeded
            value = self.value_heuristic.evaluate(state)
            node.values_estimates.append(value)
            utility = self.utility_estimator.estimate(node)
            # print('Node budget state', state, 'value', utility)
            return utility
        else:
            value = 0.0
            next_state_to_values = dict()
            next_depth = depth if node.virtual else depth -1 # skip virtual nodes

            # enumerate actors
            if node.actors is None or revise:
                node.actors = self.actor_enumerator.enumerate(state)

            # print('node actors', node.actors)

            if -1 in node.actors: # random
                assert len(node.actors) == 1

                if node.actions is None or revise:
                    node.actions = self.action_enumerator.enumerate(state, -1)

                # print('node actions', node.actions)

                if not node.action_to_next_state or revise: # Dictionary is empty
                    for action in node.actions:
                        # print('action', action)
                        node.action_to_next_state[action] = self.forward_transistor.transition(state, {-1: action})
                if not node.probs_over_actions or revise: # Dictionary is empty
                    node.probs_over_actions = self.action_predictor.predict(state, node.actions, -1)
                # print('probs over actions', node.probs_over_actions)
                for next_state in set(node.action_to_next_state.values()):
                    # print('next state', next_state)
                    next_state_to_values[next_state] = self._expand(graph, next_state, node, next_depth,
                                                                    revise=revise, oracle=oracle,
                                                                    alpha = alpha, beta = beta,
                                                                    threshold = threshold,
                                                                    node_budget=node_budget)
                
                # print('next state to values', next_state_to_values)

                # add expected value over actions 
                for action in node.actions:
                    next_state = node.action_to_next_state[action]
                    prob = node.probs_over_actions[action]
                    value += prob*next_state_to_values[next_state]

                # print('Random state', state, 'expected value', value)

            elif (0 in node.actors) and (1 in node.actors): # simultaneous
                assert len(node.actors) == 2

                # enumerate actions for each actor
                if not node.actor_to_actions or revise:
                    for actor in node.actors:
                        node.actor_to_actions[actor] = self.action_enumerator.enumerate(state, actor)

                # print('actor to actions', node.actor_to_actions)

                # print([x for x in itertools.product(*node.actor_to_actions.values())])
                        
                # create next states
                if node.next_states is None or revise:
                    node.next_states = set()

                # set proaction to value to be -inf
                for proaction in node.actor_to_actions[0]:
                    node.proaction_to_value[proaction] = -float('inf')
                
                # print('actions', node.actions) 
                maxvalue = -float('inf')
                for proaction in node.actor_to_actions[0]:
                    minvalue = float('inf')
                    for adaction in node.actor_to_actions[1]:
                        joint_action = ((0, proaction), (1, adaction))
                        next_state = self.forward_transistor.transition(state, dict(joint_action))
                        node.next_states.add(next_state)
                        if next_state not in next_state_to_values:
                            next_state_to_values[next_state] = self._expand(graph, next_state, 
                                                                           node, next_depth,
                                                                           alpha = alpha, beta = beta,
                                                                           threshold = threshold,
                                                                           revise=revise, oracle=oracle,
                                                                           node_budget=node_budget)
                        minvalue = min(minvalue, next_state_to_values[next_state])
                        if minvalue < alpha + threshold:
                            break
                        beta = min(beta, minvalue)
                    maxvalue = max(maxvalue, minvalue)
                    node.proaction_to_value[proaction] = minvalue
                    if maxvalue > beta - threshold:
                        break
                    alpha = max(alpha, maxvalue)

                # print('next states', node.next_states)

                # print('next state to values', next_state_to_values)

                # print('proaction to value', node.proaction_to_value)

                # value should be max of actions
                value = maxvalue

                # print('Simultaneous state', state, 'minimax value', value)
            
            else:
                print('node actors', node.actors)
                print('state', state)
                print('state actors', state.actors)
                raise NotImplementedError
                
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