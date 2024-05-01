from search_src.searchlight.headers import *
from search_src.searchlight.datastructures.beliefs import *
from search_src.searchlight.datastructures.graphs import ValueGraph2

import itertools
from search_src.searchlight.utils import *
from datetime import datetime

class FullSearch(Search):
    '''
    Abstract class for full search algorithms
    '''
    def __init__(self, forward_transistor: ForwardTransitor2,
                 value_heuristic: ValueHeuristic2, actor_enumerator: ActorEnumerator,
                 action_enumerator: ActionEnumerator, action_predictor: PolicyPredictor,):
        self.forward_transistor = forward_transistor
        self.value_heuristic = value_heuristic
        self.action_enumerator = action_enumerator
        self.actor_enumerator = actor_enumerator
        self.action_predictor = action_predictor

        super().__init__()

class SMMinimax(FullSearch):
    '''
    Used to perform expected minimax search

    This only works if the opponent is a single agent and the game is a zero-sum game

    TODO: change this to intermediate reward based
    '''
    def __init__(self, forward_transistor: ForwardTransitor2,
                 value_heuristic: ValueHeuristic2, actor_enumerator: ActorEnumerator,
                 action_enumerator: ActionEnumerator, action_predictor: PolicyPredictor,
                 depth=3, render = False, revise = False, oracle = True, node_budget=None):
        '''
        Args:
            forward_transistor: forward transistor
            value_heuristic: value heuristic
            actor_enumerator: actor enumerator
            action_enumerator: action enumerator
            action_predictor: action predictor
            depth: depth to expand to
            render: whether to render the graph or not
            revise: whether to revise the graph or not
            oracle: whether the opponent always plays the best response or not as if they knew the protagonist's policy
            node_budget: maximum number of nodes to expand
        '''
        
        super().__init__(forward_transistor, value_heuristic, actor_enumerator,
                         action_enumerator, action_predictor,)
        self.depth = depth
        self.render = render
        self.revise = revise
        self.oracle = oracle
        self.node_budget = node_budget

    def get_best_action(self, graph: ValueGraph2, state: State, actor=0):
        '''
        Get the best action to take for the given state
        '''
        node = graph.get_node(state)
        if node is None:
            raise ValueError('Node not found in graph')
        if actor in node.actor_to_best_action:
            return node.actor_to_best_action[actor]
        else:
            raise ValueError('Actor not found in node')
        
    def expand(self, graph: ValueGraph2, state: State,) -> None:
        '''
        Expand starting from a node
        
        Args:
            state: state to expand from
            depth: depth to expand to
            render: whether to render the graph or not
            revise: whether to revise the graph or not
            oracle: whether the opponent always plays the best response or not as if they knew the protagonist's policy
        '''
        self.nodes_expanded = 0
        self._expand(graph, state, None, self.depth, )

    def _expand(self, graph: ValueGraph2, state: State, 
                prev_node = None, depth=3,) -> dict:
        '''
        Expand starting from a node
        
        Args:
            state: state to expand from
            depth: depth to expand to
            render: whether to render the graph or not
            revise: whether to revise the graph or not
            oracle: whether the opponent always plays the best response or not as if they knew the protagonist's policy

        Returns:
            actor_to_value: mapping from actor to value
        '''

        if not self.oracle:
            raise NotImplementedError
        
        # print('Now exploring state', state)

        node = graph.get_node(state)
        if node is None: # node does not exist, create it
            node = graph.add_state(state)   
        if prev_node is not None:
            node.parents.add(prev_node)
            prev_node.children.add(node)

        # check if node is terminal
        if node.is_done():
            value = state.get_reward()
            # print('Terminal state', state, 'value', utility)
        elif depth == 0:
            value, opponent_value = self.value_heuristic.evaluate(state)
            # print('Depth 0 state', state, 'value', utility)
        elif self.node_budget is not None and self.nodes_expanded >= self.node_budget:
            # use heuristic to estimate value if node budget is exceeded
            value, opponent_value = self.value_heuristic.evaluate(state)
        else:
            # note that this would always be an unexpected node, unlike MCTS
            self.total_nodes_expanded += 1
            self.nodes_expanded += 1

            value = 0.0
            next_state_to_values = dict()
            next_depth = depth if node.virtual else depth -1 # skip virtual nodes

            # enumerate actors
            node.actors = self.actor_enumerator.enumerate(state)

            # print('node actors', node.actors)

            if -1 in node.actors: # random
                assert len(node.actors) == 1

                env_actions = self.action_enumerator.enumerate(state, -1)
                # joint actions
                node.actions = {((-1, action),) for action in env_actions}


                # print('node actions', node.actions)

                next_states = set()
                for action in node.actions:
                    # print('action', action)
                    next_state = self.forward_transistor.transition(state, dict(action))
                    node.action_to_next_state[action] = next_state
                    next_states.add(next_state)
                
                probs = self.action_predictor.predict(state, env_actions, -1)
                action_to_prob = {action: prob for action, prob in zip(node.actions, probs)}
                node.actor_to_action_to_prob  = {-1: action_to_prob}
                
                # print('probs over actions', node.probs_over_actions)
                for next_state in set(node.action_to_next_state.values()):
                    # print('next state', next_state)
                    next_state_to_values[next_state] = self._expand(graph, next_state, node, next_depth,)
                
                # print('next state to values', next_state_to_values)

                # add expected value over actions 
                for action in node.actions:
                    next_state = node.action_to_next_state[action]
                    prob = action_to_prob[action]
                    value += prob*next_state_to_values[next_state]

                # print('Random state', state, 'expected value', value)

            elif (0 in node.actors) and (1 in node.actors): # simultaneous
                assert len(node.actors) == 2

                # enumerate actions for each actor
                actor_to_actions = dict()
                for actor in node.actors:
                    actor_to_actions[actor] = self.action_enumerator.enumerate(state, actor)

                # print('actor to actions', actor_to_actions)

                # print([x for x in itertools.product(*node.actor_to_actions.values())])
            
                # enumerate all possible joint actions as tuples of tuples (actor,action) pairs
                node.actions = [tuple(zip(node.actors, x)) for x in itertools.product(*actor_to_actions.values())]
                # node.actions = dict_to_set_of_cartesian_products(node.actions)
                
                # print('actions', node.actions)

                # find next states
                next_states = set()
                # print('next states not found')
                for joint_action in node.actions:
                    next_state = self.forward_transistor.transition(state, dict(joint_action))
                    next_states.add(next_state)
                    node.action_to_next_state[joint_action] = next_state
                    # print('joint action', joint_action, 'next state', next_state)
                next_states = frozenset(next_states)

                # print('next states', node.next_states)
                
                # print('nodes expanded simult', self.nodes_expanded)

                # # if len(node.next_states) > node_budget-self.nodes_expanded and node_budget is not None, then use heuristic
                # if node_budget is not None and len(node.next_states) > node_budget-self.nodes_expanded:
                #     # use heuristic to estimate value if node budget is exceeded
                #     value = self.value_heuristic.evaluate(state)
                #     # print('Simultaneous state', state, 'heuristic value', value)
                #     node.values_estimates.append(value)
                #     utility = self.utility_estimator.estimate(node)
                #     return utility

                # expand next states
                for next_state in next_states:
                    # print('next state', next_state)
                    # print('next state hash', hash(next_state))
                    next_state_to_values[next_state] = self._expand(graph, next_state, node, next_depth,)

                # print('next state to values', next_state_to_values)

                # for each protagonist action, find the best response of the opponent
                opponent_best_responses = dict() # mapping from protagonist action to tuple of (opponent action, value)

                # set br values to infinity
                for proaction in actor_to_actions[0]:
                    opponent_best_responses[proaction] = (None, float('inf'))

                # find best response for each protagonist action
                for joint_action in node.actions:
                    proaction = dict(joint_action)[0]
                    value = next_state_to_values[node.action_to_next_state[joint_action]]
                    # print('joint action', joint_action, 'value', value)
                    if value < opponent_best_responses[proaction][1]:
                        opponent_best_responses[proaction] = (joint_action[1], value)

                # print('opponent best responses', opponent_best_responses)

                proaction_to_value = dict() # mapping from protagonist action to value
                # set action to value to be opponent best response
                for proaction in actor_to_actions[0]:
                    proaction_to_value[proaction] = opponent_best_responses[proaction][1]

                # value should be max of actions
                best_action, value = max(proaction_to_value.items(), key=lambda x: x[1])

                node.actor_to_best_action[0] = best_action

                # print('Simultaneous state', state, 'minimax value', value)
            
            else:
                print('node actors', node.actors)
                print('state', state)
                # print('state actors', state.actors)
                raise NotImplementedError
                
            if self.render and not node.virtual:
                plt = graph.to_mathplotlib()
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'Search/output/output_graph_{timestamp}.png'
                plt.savefig(filename)
                plt.close()
                # plt.show()
        
        # print('value', value)
        node.actor_to_value_estimates[0].append(value)
        node.actor_to_value_estimates[1].append(-value)
        utility = graph.get_estimated_value(node, 0)
        return utility
        
# class SMAlphaBetaMinimax(FullSearch):
#     '''
#     Used to perform simultaneous expected minimax search with alpha-beta pruning

#     This only works if the opponent is a single agent and the game is a zero-sum game
#     '''
#     def __init__(self, forward_transistor: ForwardTransitor,
#                  value_heuristic: ValueHeuristic, actor_enumerator: ActorEnumerator,
#                  action_enumerator: ActionEnumerator, action_predictor: ActionPredictor,
#                  utility_estimator: UtilityEstimator):
        
#         super().__init__(forward_transistor, value_heuristic, actor_enumerator,
#                          action_enumerator, action_predictor, utility_estimator)
        
#     def expand(self, graph: ValueGraph2, state: State,
#                 prev_node = None, depth=3, render = False, 
#                 revise = False, oracle = True, 
#                 alpha = -float('inf'), beta = float('inf'), threshold = 0.0,
#                 node_budget=None):
#           '''
#           Expand starting from a node
          
#           Args:
#                 state: state to expand from
#                 depth: depth to expand to
#                 render: whether to render the graph or not
#                 revise: whether to revise the graph or not
#                 oracle: whether the opponent always plays the best response or not as if they knew the protagonist's policy
#                 alpha: alpha value for alpha-beta pruning
#                 beta: beta value for alpha-beta pruning
#                 threshold: threshold for alpha-beta pruning
    
#           Returns:
#                 value: updated value of the node
#           '''
#           self.nodes_expanded = 0
#           return self._expand(graph, state, prev_node, depth, render, revise, oracle, alpha, beta, threshold, node_budget)

#     def _expand(self, graph: ValueGraph, state: State, 
#                prev_node = None, depth=3, render = False, 
#                revise = False, oracle = True, 
#                alpha = -float('inf'), beta = float('inf'), threshold = 0.0,
#                node_budget=None):
#         '''
#         Expand starting from a node
        
#         Args:
#             state: state to expand from
#             depth: depth to expand to
#             render: whether to render the graph or not
#             revise: whether to revise the graph or not
#             oracle: whether the opponent always plays the best response or not as if they knew the protagonist's policy
#             alpha: alpha value for alpha-beta pruning
#             beta: beta value for alpha-beta pruning
#             threshold: threshold for alpha-beta pruning

#         Returns:
#             value: updated value of the node
#         '''

#         # self.total_nodes_expanded += 1
#         # self.nodes_expanded += 1

#         if not oracle:
#             raise NotImplementedError
        
#         # print('Now exploring state', state)

#         node = graph.get_node(state)
#         if node is None: # node does not exist, create it
#             node = graph.add_state(state)   
#         if prev_node is not None:
#             node.parents.add(prev_node)
#             prev_node.children.add(node)

#         # check if node is terminal
#         if state.is_done():
#             value = state.get_reward()
#             # print('Terminal state', state, 'value', utility)
#         elif depth == 0:
#             value = self.value_heuristic.evaluate(state)
#             # print('Depth 0 state', state, 'value', utility)
#         elif node_budget is not None and self.nodes_expanded >= node_budget:
#             # use heuristic to estimate value if node budget is exceeded
#             value = self.value_heuristic.evaluate(state)
#         else:
#             value = 0.0
#             self.total_nodes_expanded += 1
#             self.nodes_expanded += 1
#             next_state_to_values = dict()
#             next_depth = depth if node.virtual else depth -1 # skip virtual nodes

#             # enumerate actors
#             if node.actors is None or revise:
#                 node.actors = self.actor_enumerator.enumerate(state)

#             # print('node actors', node.actors)

#             if -1 in node.actors: # random
#                 assert len(node.actors) == 1

#                 if node.actions is None or revise:
#                     node.actions = self.action_enumerator.enumerate(state, -1)

#                 # print('node actions', node.actions)

#                 if not node.action_to_next_state or revise: # Dictionary is empty
#                     node.next_states = set()
#                     for action in node.actions:
#                         # print('action', action)
#                         next_state = self.forward_transistor.transition(state, {-1: action})
#                         node.action_to_next_state[action] = next_state
#                         node.next_states.add(next_state)
#                 if not node.probs_over_actions or revise: # Dictionary is empty
#                     node.probs_over_actions = self.action_predictor.predict(state, node.actions, -1)

                

#                 # if len(node.next_states) > node_budget-self.nodes_expanded and node_budget is not None, then use heuristic
#                 # if node_budget is not None and len(node.next_states) > node_budget-self.nodes_expanded:
#                 #     # use heuristic to estimate value if node budget is exceeded
#                 #     value = self.value_heuristic.evaluate(state)
#                 #     # print('Simultaneous state', state, 'heuristic value', value)
#                 #     node.values_estimates.append(value)
#                 #     utility = self.utility_estimator.estimate(node)
#                 #     return utility
#                 # print('probs over actions', node.probs_over_actions)
#                 for next_state in set(node.action_to_next_state.values()):
#                     # print('next state', next_state)
#                     next_state_to_values[next_state] = self._expand(graph, next_state, node, next_depth,
#                                                                     revise=revise, oracle=oracle,
#                                                                     alpha = alpha, beta = beta,
#                                                                     threshold = threshold,
#                                                                     node_budget=node_budget)
                
#                 # print('next state to values', next_state_to_values)

#                 # add expected value over actions 
#                 for action in node.actions:
#                     next_state = node.action_to_next_state[action]
#                     prob = node.probs_over_actions[action]
#                     value += prob*next_state_to_values[next_state]

#                 # print('Random state', state, 'expected value', value)

#             elif (0 in node.actors) and (1 in node.actors): # simultaneous
#                 assert len(node.actors) == 2

#                 # enumerate actions for each actor
#                 if not node.actor_to_actions or revise:
#                     for actor in node.actors:
#                         node.actor_to_actions[actor] = self.action_enumerator.enumerate(state, actor)

#                 # print('actor to actions', node.actor_to_actions)

#                 # print([x for x in itertools.product(*node.actor_to_actions.values())])
                        
#                 # # create next states
#                 # if node.next_states is None or revise:
#                 #     node.next_states = set()

#                 # set proaction to value to be -inf
#                 for proaction in node.actor_to_actions[0]:
#                     node.proaction_to_value[proaction] = -float('inf')

#                 # find next states
#                 if node.next_states is None or revise:
#                     node.next_states = set()
#                     # print('next states not found')
#                     for proaction in node.actor_to_actions[0]:
#                         for adaction in node.actor_to_actions[1]:
#                             joint_action = ((0, proaction), (1, adaction))
#                             next_state = self.forward_transistor.transition(state, dict(joint_action))
#                             node.next_states.add(next_state)
#                             node.action_to_next_state[joint_action] = next_state
#                             # print('joint action', joint_action, 'next state', next_state)

#                 # if len(node.next_states) > node_budget-self.nodes_expanded and node_budget is not None, then use heuristic
#                 # if node_budget is not None and len(node.next_states) > node_budget-self.nodes_expanded:
#                 #     # use heuristic to estimate value if node budget is exceeded
#                 #     value = self.value_heuristic.evaluate(state)
#                 #     # print('Simultaneous state', state, 'heuristic value', value)
#                 #     node.values_estimates.append(value)
#                 #     utility = self.utility_estimator.estimate(node)
#                 #     return utility
                
#                 # print('actions', node.actions) 
#                 maxvalue = -float('inf')
#                 for proaction in node.actor_to_actions[0]:
#                     minvalue = float('inf')
#                     for adaction in node.actor_to_actions[1]:
#                         joint_action = ((0, proaction), (1, adaction))
#                         next_state = node.action_to_next_state[joint_action]
#                         if next_state not in next_state_to_values:
#                             next_state_to_values[next_state] = self._expand(graph, next_state, 
#                                                                            node, next_depth,
#                                                                            alpha = alpha, beta = beta,
#                                                                            threshold = threshold,
#                                                                            revise=revise, oracle=oracle,
#                                                                            node_budget=node_budget)
#                         minvalue = min(minvalue, next_state_to_values[next_state])
#                         if minvalue < alpha + threshold:
#                             break
#                         beta = min(beta, minvalue)
#                     maxvalue = max(maxvalue, minvalue)
#                     node.proaction_to_value[proaction] = minvalue
#                     if maxvalue > beta - threshold:
#                         break
#                     alpha = max(alpha, maxvalue)

#                 # print('next states', node.next_states)

#                 # print('next state to values', next_state_to_values)

#                 # print('proaction to value', node.proaction_to_value)

#                 # value should be max of actions
#                 value = maxvalue

#                 # print('Simultaneous state', state, 'minimax value', value)
            
#             else:
#                 print('node actors', node.actors)
#                 print('state', state)
#                 print('state actors', state.actors)
#                 raise NotImplementedError
                
#             if render and not node.virtual:
#                 plt = graph.to_mathplotlib()
#                 timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#                 filename = f'Search/output/output_graph_{timestamp}.png'
#                 plt.savefig(filename)
#                 plt.close()
#                 # plt.show()
            
#         node.values_estimates.append(value)
#         utility = self.utility_estimator.estimate(node)
#         return utility