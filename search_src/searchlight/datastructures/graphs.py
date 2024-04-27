from ..headers import *
from .beliefs import ValueNode2, Graph
from .adjusters import QValueAdjuster
from .estimators import UtilityEstimator, UtilityEstimatorLast

import networkx as nx
# import pygraphviz as pgv
from networkx.drawing.nx_agraph import to_agraph
import matplotlib.pyplot as plt
import time 
import matplotlib.colors as mcolors
import numpy as np
from typing import List, Tuple, Set, Any, Optional
from collections import defaultdict

class ValueGraph2(Graph):
    '''
    A DAG where each node represents a state and each edge represents an action

    Updated version of ValueGraph

    TODO: fix graph so that value estimates contains values for each actor
    '''
    id_to_node: dict[State, ValueNode2]

    def __init__(self, players, adjuster: Optional[QValueAdjuster] = None, utility_estimator: Optional[UtilityEstimator] = None, 
                 rng = np.random.default_rng()):
        super().__init__()
        if adjuster is None:
            adjuster = QValueAdjuster()
        if utility_estimator is None:
            utility_estimator = UtilityEstimatorLast()
        self.adjuster = adjuster
        self.utility_estimator = utility_estimator
        self.rng = rng
        self.players = players # set of players we need to keep track of values for

        # self.boltzmann_policy_temperature = 1.0
        self.boltzmann_policy_scheduler_constant = 1.0 # set this to the variance of the rewards for optimal temperature

        # create an end node
        self.end_node = ValueNode2(END_STATE, set(), set())

    def get_node(self, id)-> Optional[ValueNode2]:
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
        
    def get_end_node(self):
        '''
        Returns the end node
        '''
        return self.end_node
    
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
        
    def add_edge(self, parent: State, child: State,) -> None:
        '''
        Adds an edge to the graph

        Args:
            parent: parent state
            child: child state
            action: action taken to get from parent to child

        Returns:
            None
        '''
        parent_node = self.get_node(parent)
        child_node = self.get_node(child)
        if parent_node is None:
            parent_node = self.add_state(parent)
        parent_node.children.add(child_node)
        if child_node is None:
            child_node = self.add_state(child)
        child_node.parents.add(parent_node)
        
    # def get_best_action(self, state: State) -> dict:
    #     '''
    #     NOTE: this is part of search now
    #     Returns dictionary of best actions by actor for a given state
    #     '''
    #     node = self.get_node(state) 
    #     return dict(self.select_action(node))
        
    def simulate_trajectory(self, state: State, num_steps: int = -1) -> List[Tuple[State, Tuple[Tuple[Any, Any], ...]]]:
        '''
        Simulates a trajectory from the given state for a given number of steps.
        Players will take optimal actions according to the graph
        Environment will take actions according to the random policy

        Args:
            state: state to simulate from
            num_steps: number of steps to simulate. if -1, simulates until the end of the graph (unexpanded node)

        Returns:
            trajectory: list of tuples of state and joint action. will be of the form [(state, joint_action), ...]. At terminal states, the joint action will be None
        '''
        trajectory = []

        # Initialize a variable to keep track of the current step number.
        current_step = 0

        while True:
            node = self.get_node(state)

            # Check if the node is not expanded (i.e., node is None) or if the node is terminal.
            # In either case, append the current state with None action and break.
            if node is None or node.is_done():
                trajectory.append((state, None))
                break

            joint_action = self.select_action(node)
            next_state = node.action_to_next_state[joint_action]
            trajectory.append((state, joint_action))
            state = next_state

            current_step += 1

            # If num_steps is not -1 (fixed step case), check if the current step equals num_steps.
            # Break if we've reached the required number of steps.
            if num_steps != -1 and current_step >= num_steps:
                break

        return trajectory

    def get_joint_action_probabilities(self, node: ValueNode2) -> dict:
        '''
        Returns the joint action probabilities for a given state
        '''
        action_to_prob = dict()
        for joint_action in node.get_joint_actions():
            prob = 1.0
            for actor, action in joint_action:
                prob *= node.actor_to_action_to_prob[actor][action]
            action_to_prob[joint_action] = prob
        return action_to_prob

    def select_action(self, node: ValueNode2) -> Tuple[Tuple[Any, Any], ...]:
        '''
        Selects a joint action to simulate

        Returns empty tuple if terminal state
        '''
        # calculate the probability of each joint action
        action_to_prob = self.get_joint_action_probabilities(node)

        # get action of each actor using _select_action_by_actor. put in tuple of tuples (actor, action)
        joint_action = tuple((actor, self.select_action_by_actor(node, actor, action_to_prob)) for actor in node.get_actors())
        return joint_action
        
    def select_action_by_actor(self, node: ValueNode2, actor, action_to_prob: dict):
        '''
        Selects an action to simulate for a given actor
        If no actions are available, returns None

        Args:
            node: node to select action from
            actor: actor to select action for
            action_to_prob: dictionary of estimated joint actions to probability
        '''

        # if there are no actions available, return None
        # if not node.actor_to_action_to_prob[actor]:
        #     return None
        
        # if actor == -1 (environment), then select an action according to the policy
        # i.e. probability weights should be node.actor_to_action_to_prob[actor].values()
        if actor == -1:
            return self.rng.choice(list(node.actor_to_action_to_prob[actor].keys()), 
                                   p = list(node.actor_to_action_to_prob[actor].values()))
        # otherwise choose the action with the highest adjusted qvalue
        else:
            adj_qvalues = self.get_adjusted_qvalues(node, actor, action_to_prob) # this is a dictionary of action to qvalue
            
            return max(adj_qvalues, key=adj_qvalues.get)
        
    def get_qvalues(self, node: ValueNode2, actor, action_to_prob: dict) -> dict[Any, float]:
        '''
        Gets the qvalues for a given actor
        '''
        # find the expected qvalue for each action
        action_to_expected_qvalue = dict()
        # action_to_expected_qvalue to 0 for each action
        for action in node.actor_to_action_to_prob[actor].keys():
            action_to_expected_qvalue[action] = 0.0
        # find expected qvalue by summing over all joint actions that contain the action
        for joint_action, prob in action_to_prob.items():
            # get the action in the joint action that corresponds to the actor
            action = dict(joint_action)[actor]
            # get next state
            next_state = node.action_to_next_state[joint_action]

            next_node = self.get_node(next_state)
            if next_node is None:
                # TODO: this is a hack, fix it
                next_node = self.add_state(next_state)

            # TODO: we forgot to check if the state is terminal here. if terminal, may not have value estimates
            # get the value of the next state using utility estimator
            next_value = self.get_estimated_value(next_node , actor)
            # add the value to the expected qvalue
            action_to_expected_qvalue[action] += prob*next_value
            # get the intermediate reward
            reward = node.action_to_actor_to_reward[joint_action][actor]
            # add the reward to the expected qvalue
            action_to_expected_qvalue[action] += prob*reward
        return action_to_expected_qvalue
            
    def get_adjusted_qvalues(self, node: ValueNode2, actor, action_to_prob: dict) -> dict[Any, float]:
        '''
        Gets the adjusted qvalues for a given actor.
        Usually the PUCT value is used

        Args:
            node: node to get the adjusted qvalues for
            actor: actor to get the adjusted qvalues for
            action_to_prob: dictionary of estimated joint actions to probability

        Returns:
            action_to_value: dictionary of adjusted qvalues
        '''

        # get the expected qvalue for each action
        action_to_expected_qvalue = self.get_qvalues(node, actor, action_to_prob)
        

        # print out the state, node.actor_to_action_to_prob
        # print('state', node.state)
        # print('node.actor_to_action_to_prob', node.actor_to_action_to_prob)
        # print('node.actor_to_action_to_prob[actor].keys()', node.actor_to_action_to_prob[actor].keys())

        # for each action, use the adjuster to get the adjusted qvalue
        action_to_value = dict()
        for action in node.actor_to_action_to_prob[actor].keys():
            action_to_value[action] = self.adjuster.adjust(action_to_expected_qvalue[action],
                                                        node.actor_to_action_to_prob[actor][action],
                                                        node.visits, node.actor_to_action_visits[actor][action])
        return action_to_value
    
    def get_boltzmann_policy(self, node: ValueNode2, actor, temperature: float, adjusted_q=False) -> dict:
        '''
        Gets the boltzmann policy for a given actor

        Args:
            node: node to get the policy for
            actor: actor to get the policy for
            temperature: temperature parameter
            adjusted_q: whether to use adjusted qvalues or not

        Returns:
            policy: dictionary of actions to probabilities
        '''
        action_to_prob = self.get_joint_action_probabilities(node)
        # get the qvalues for the actor
        if adjusted_q:
            action_to_qvalue = self.get_adjusted_qvalues(node, actor, action_to_prob)
        else:
            action_to_qvalue = self.get_qvalues(node, actor, action_to_prob)
        # calculate the boltzmann distribution
        policy = dict()
        for action in action_to_qvalue:
            policy[action] = np.exp(action_to_qvalue[action]/temperature)
        # normalize the policy
        total = sum(policy.values())
        for action in policy:
            policy[action] /= total
        return policy
    
    def get_scheduled_boltzmann_policy(self, node: ValueNode2, actor, adjusted_q = False) -> dict:
        '''
        Gets the boltzmann policy for a given actor but with scheduled temperature

        Args:
            node: node to get the policy for
            actor: actor to get the policy for
            adjusted_q: whether to use adjusted qvalues or not

        Returns:
            policy: dictionary of actions to probabilities
        '''
        constant = self.boltzmann_policy_scheduler_constant
        actions = node.get_actions_for_actor(actor)
        action_to_temperature = dict()
        for action in actions:
            action_to_temperature[action] = np.sqrt(constant/node.actor_to_action_visits[actor][action])

        action_to_prob = self.get_joint_action_probabilities(node)

        # get the qvalues for the actor
        if adjusted_q:
            action_to_qvalue = self.get_adjusted_qvalues(node, actor, action_to_prob)
        else:
            action_to_qvalue = self.get_qvalues(node, actor, action_to_prob)

        # calculate the boltzmann distribution
        policy = dict()
        for action in action_to_qvalue:
            policy[action] = np.exp(action_to_qvalue[action]/action_to_temperature[action])
        # normalize the policy
        total = sum(policy.values())
        for action in policy:
            policy[action] /= total
        return policy

    def get_estimated_value(self, node: ValueNode2, actor) -> float:
        '''
        Gets the estimated value of a node for a given actor

        Args:
            node: node to get the estimated value of
            actor: actor to get the estimated value for

        Returns:
            value: estimated value of the node for the actor
        '''
        return self.utility_estimator.estimate(node, actor)
    
    def get_highest_value_state(self, actor):
        '''
        Gets the state with the highest estimated value for a given actor

        Args:
            actor: actor to get the highest value state for

        Returns:
            state: state with the highest value for the actor
        '''
        return max(self.id_to_node.values(), key=lambda node: self.get_estimated_value(node, actor)).state

    

    def backpropagate(self, node: ValueNode2, action, next_state: State):
        '''
        Backpropagates the value from the next state to the current node for each actor

        Args:
            node: node to backpropagate to
            action: action taken to get to the next state
            next_state: next state to backpropagate from
            graph: graph to backpropagate to

        Returns:
            None
        '''
        # print('next_state', next_state)
        # print('state', node.state)

        # we need to backpropagate for each player in self.players
        for actor in self.players:
            next_node = self.get_node(next_state)
            if next_node is None:
                next_node = self.add_state(next_state)
                # TODO: this is a hack, fix it

            # get the value of the next state
            next_value = self.utility_estimator.estimate(next_node, actor)
            # get the intermediate reward
            reward = node.action_to_actor_to_reward[action][actor]
            # update the value of the current node


            # NOTE: this is not expected value, but the actual value
            node.actor_to_value_estimates[actor].append(next_value + reward)
            
        # update the number of visits to the state
        node.visits += 1
        
        # update the number of visits to the state-action pair
        for actor in node.get_actors():
            # get the action that this actor took
            actor_action = dict(action)[actor]
            # update the number of visits to the state-action pair
            node.actor_to_action_visits[actor][actor_action] += 1

    def to_networkx(self):
        '''
        Returns the graph as a networkx graph, with values as node.values
        '''
        G = nx.DiGraph()
        for node in self.id_to_node.values():
            value = self.utility_estimator.estimate(node, 0)
            # round value to 4 significant figures
            value = round(value, 4)
            visits = node.visits
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
    
class PartialValueGraph(ValueGraph2):
    '''
    A graph that also keeps track of information sets
    Each information set contains a set of nodes (states) that are indistinguishable to the player
    Whenever we take qvalues, we take the (weighted?) average of the qvalues of the nodes in the information set since the player cannot distinguish between them
    We can estimate the frequency of each node in the information set by the number of visits to the information set (MC estimate)
    '''

    player_to_information_set_to_set_of_nodes: dict[Any, dict[Any, Set[ValueNode2]]] 
    # player_and_node_to_information_set: dict[Tuple[Any, ValueNode2], Any]

    def __init__(self, players, adjuster: Optional[QValueAdjuster] = None, utility_estimator: Optional[UtilityEstimator] = None, rng = np.random.default_rng(),):
        super().__init__(players=players, adjuster=adjuster, utility_estimator=utility_estimator, rng=rng)
        self.player_to_information_set_to_set_of_nodes = defaultdict(dict)
        # self.player_and_node_to_information_set = dict()
        self.logger.info(f'PartialValueGraph initialized with players {players}')

    def get_information_set(self, node: ValueNode2, actor) -> Any:
        '''
        Gets the information set of a state for a given actor
        '''
        # return self.player_and_node_to_information_set[(actor, node)]
        return node.state.get_information_set(actor)
    
    def add_state(self, state: HiddenState, parent_states=[], child_states=[]):
        '''
        Adds a state to the tree. Will also add the state to the information set of the player

        Args:
            state: state to add

        Returns:
            node: node corresponding to the state added
        '''
        # self.logger.info(f'Adding state {state} to the graph')
        parents = set([self.id_to_node[parent_state] for parent_state in parent_states])
        children = set([self.id_to_node[child_state] for child_state in child_states])
        if state not in self.id_to_node:
            # TODO: should be generalized, test if works
            node = ValueNode2(state, parents, children)
            self.id_to_node[state] = node

            # add the state to the information set for all players
            for player in self.players:
                information_set = self.get_information_set(node, player)
                if information_set not in self.player_to_information_set_to_set_of_nodes[player]:
                    self.player_to_information_set_to_set_of_nodes[player][information_set] = set()
                self.player_to_information_set_to_set_of_nodes[player][information_set].add(node)
                
                # log the information set added for debugging
                # self.logger.info(f'Information set {information_set} added for player {player}')
                
                # add the node to the information set
                # self.player_and_node_to_information_set[(player, node)] = information_set

            return node
        else:
            raise ValueError(f"state {state} already exists in the graph")

    def get_qvalues(self, node: ValueNode2, actor, action_to_prob: dict) -> dict[Any, float]:
        '''
        Gets the qvalues for a given actor
        '''
        # find the expected qvalue for each action
        action_to_expected_qvalue = dict()
        # action_to_expected_qvalue to 0 for each action
        for action in node.actor_to_action_to_prob[actor].keys():
            action_to_expected_qvalue[action] = 0.0
        # find expected qvalue by summing over all joint actions that contain the action
        for joint_action, prob in action_to_prob.items():
            # get the action in the joint action that corresponds to the actor
            action = dict(joint_action)[actor]

            # get the information set of the node
            information_set = self.get_information_set(node, actor)

            # get the nodes in the information set
            nodes = self.player_to_information_set_to_set_of_nodes[actor][information_set]

            value_list = [] # TODO: this is a hack, the ordering of the estimates will be wrong, so utility estimator last will not work
            for node1 in nodes:
                # get the next state
                next_state = node1.action_to_next_state[joint_action]

                next_node = self.get_node(next_state)
                if next_node is None:
                    next_node = self.add_state(next_state)
                    self.logger.warning(f'Node {next_state} not found in graph')
                    raise ValueError('Node not found in graph') # TODO: this should never happen
                    # TODO: this should never happen

                # self.logger.info(f'Getting qvalues for state {next_node.state} for actor {actor}')

                # # log whether the state is terminal
                # self.logger.info(f'Node {next_node.state} is terminal: {next_node.is_done()}')

                # if the node is not terminal, add the value estimates of the next node to the value_list
                if not next_node.is_done():
                    value_list.extend(next_node.actor_to_value_estimates[actor])

            # get the value of the next state using utility estimator
            next_value = self.utility_estimator.estimate_list(value_list)
            # add the value to the expected qvalue
            action_to_expected_qvalue[action] += prob*next_value
            # get the intermediate reward
            reward = node.action_to_actor_to_reward[joint_action][actor]
            # add the reward to the expected qvalue
            action_to_expected_qvalue[action] += prob*reward
        return action_to_expected_qvalue