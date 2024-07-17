from ..headers import *
from .beliefs import ValueNode, Graph, InformationSetNode, ValueNode2
from .adjusters import QValueAdjuster, PUCTAdjuster
from .estimators import UtilityEstimator, UtilityEstimatorLast, UtilityEstimatorMean

import networkx as nx
# import pygraphviz as pgv
from networkx.drawing.nx_agraph import to_agraph
import matplotlib.pyplot as plt
import time 
import matplotlib.colors as mcolors
import numpy as np
from typing import List, Tuple, Set, Any, Optional, Callable, Union
from collections import defaultdict

class ValueGraph(Graph):
    '''
    A DAG where each node represents a state and each edge represents an action

    Updated version of ValueGraph

    TODO: fix graph so that value estimates contains values for each actor
    '''
    id_to_node: dict[Hashable, ValueNode]

    def __init__(self, players, adjuster: Union[QValueAdjuster, str] = "no adjust", utility_estimator: Union[UtilityEstimator, str] = "mean", rng = np.random.default_rng()):
        super().__init__()
        if isinstance(adjuster, str):
            if adjuster == "no adjust":
                adjuster = QValueAdjuster()
            elif adjuster == "puct":
                adjuster = PUCTAdjuster()
            else:
                raise ValueError(f"adjuster {adjuster} not recognized")
        if isinstance(utility_estimator, str):
            if utility_estimator == "mean":
                utility_estimator = UtilityEstimatorMean()
            elif utility_estimator == "last":
                utility_estimator = UtilityEstimatorLast()
            else:
                raise ValueError(f"utility_estimator {utility_estimator} not recognized")
        self.adjuster = adjuster
        self.utility_estimator = utility_estimator
        self.rng = rng
        self.players = players # set of players we need to keep track of values for

        # self.boltzmann_policy_temperature = 1.0
        self.boltzmann_policy_scheduler_constant = 1.0 # set this to the variance of the rewards for optimal temperature

        # create an end node
        # self.end_node = ValueNode(END_STATE, set(), set())

    def get_node(self, id)-> Optional[ValueNode]:
        '''
        Returns the node corresponding to the id

        Args:
            id: id to get node of

        Returns:
            node: node corresponding to the id, or None if it does not exist
        '''
        if id not in self.id_to_node:
            print("Id not in graph")
            return None
        else:
            print("Id to node is: ", self.id_to_node[id])
            return self.id_to_node[id]
        
    def get_all_nodes(self) -> List[ValueNode]:
        '''
        Returns all nodes in the graph
        '''
        return list(self.id_to_node.values())
        
    # def get_end_node(self):
    #     '''
    #     Returns the end node
    #     '''
    #     return self.end_node
    
    def add_state(self, state: Hashable, parent_states=[], child_states=[]):
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
            node = ValueNode(state, parents, children)
            self.id_to_node[state] = node
            return node
        else:
            raise ValueError(f"state {state} already exists in the graph")
        
    def simulate_trajectory(self, state: Hashable, num_steps: int = -1, stop_condition: Union[Callable[[ValueNode], bool], str] = lambda x : False) -> List[Tuple[Hashable, Hashable]]:
        '''
        Simulates a trajectory from the given state for a given number of steps.
        Players will take optimal actions according to the graph
        Environment will take actions according to the random policy

        Args:
            state: state to simulate from
            num_steps: number of steps to simulate. if -1, simulates until the end of the graph (unexpanded node)

        Returns:
            trajectory: list of tuples of state and action. will be of the form [(None, start_state), (action1, state1) ...]. 
        '''
        # if initial state not in graph, return empty trajectory
        if state not in self.id_to_node:
            return []

        if isinstance(stop_condition, str):
            if stop_condition == "has_unvisited_actions":
                # check if x.get_unvisited_actions() is not empty
                stop_condition = lambda x : len(x.get_unvisited_actions()) >= 0
            else:
                raise ValueError(f"stop_condition {stop_condition} not recognized")

        trajectory: List[Tuple[Hashable, Hashable]] = [(None, state)]

        # Initialize a variable to keep track of the current step number.
        current_step = 0

        # add initial state to graph
        # if state not in self.id_to_node:
        #     self.add_state(state)
        while True:
            node = self.get_node(state)

            # Check if the node is not expanded (i.e., node is None) or if the node is terminal.
            # In either case, append the current state with None action and break.
            if node is None or node.is_done() or stop_condition(node):
                break
            else:
                print("Not stop!!!!!!!!!!")

            # print("actions at node", node.actions)
            # print("actions in prob", node.action_to_prob_weights.keys())
            # print("actions in next state", node.action_to_next_state.keys())
            action = self.select_action(node)
            next_state = node.action_to_next_state[action]
            trajectory.append((action, next_state))
            state = next_state

            current_step += 1

            # If num_steps is not -1 (fixed step case), check if the current step equals num_steps.
            # Break if we've reached the required number of steps.
            if num_steps != -1 and current_step >= num_steps:
                break
        return trajectory

    def select_action(self, node: ValueNode, maximizing_method: str = "max", filter_out_states: list[Hashable] = [], filter_out_actions: list[Hashable] = []) -> Hashable:
        '''
        Selects action to simulate. NOTE: we only select visited actions!!!

        Args:
            node: node to select the action from
            maximizing_method: method to use to select the action. Currently only "max" is supported
            filter_out_states: states to filter out
            filter_out_actions: actions to filter out

        Returns None if the node is terminal or if there are no valid actions
        '''
        if node.is_done():
            print("Node is done")
            return None
        
        # _action_to_prob = node.get_action_to_probs()
        action_to_prob = node.get_action_to_probs() 
        # action_to_prob = dict()
        # actions = node.get_actions()
        # for action in actions:
        #     action_to_prob[action] = _action_to_prob[action]
        actor = node.get_actor()
        # if actor == -1 (environment), then select an action according to the policy
        # i.e. probability weights should be node.actor_to_action_to_prob[actor].values()
        if actor == -1:
            # set the probability weights of any actions in filter_out_actions to 0
            for action in filter_out_actions:
                action_to_prob[action] = 0
            # set the probability weights of any actions that lead to states in filter_out_states to 0
            for action, next_state in node.action_to_next_state.items():
                if next_state in filter_out_states:
                    action_to_prob[action] = 0
            # renormalize the probabilities
            total = sum(action_to_prob.values())
            action_to_prob = {action: prob/total for action, prob in action_to_prob.items()}
            return self.rng.choice(list(action_to_prob.keys()), p=list(action_to_prob.values()))
        # otherwise choose the action with the highest adjusted qvalue
        else:
            adj_qvalues = self.get_adjusted_qvalues(node.state) # this is a dictionary of action to qvalue
            # print("adj qvalues", adj_qvalues)
            # filter out any actions that are in filter_out_actions
            adj_qvalues = {action: qvalue for action, qvalue in adj_qvalues.items() if action not in filter_out_actions}
            # filter out any actions that lead to states in filter_out_states
            adj_qvalues = {action: qvalue for action, qvalue in adj_qvalues.items() if node.action_to_next_state[action] not in filter_out_states}
            if maximizing_method == "max":
                return max(adj_qvalues, key=adj_qvalues.get)
            else:
                raise ValueError(f"maximizing_method {maximizing_method} not recognized")

    def select_random_unvisited_action(self, state: Hashable) -> Hashable:
        '''
        Selects a random unvisited action from the state uniformly and returns the action and nextstate. Return None if all actions are visited

        Args:
            state: state to select the action from

        Returns:
            action: selected action
        '''
        node = self.get_node(state)
        if node is None:
            raise ValueError('Node not found in graph')
        unvisited_actions = node.get_unvisited_actions()
        if len(unvisited_actions) == 0:
            return None
        action = self.rng.choice(list(unvisited_actions))
        return action
        
    def get_qvalues(self, state: Hashable,) -> dict[Any, float]:
        '''
        Gets the qvalues for the acting actor in the node
        '''
        node = self.get_node(state)
        if node is None:
            raise ValueError('Node not found in graph')
        actor = node.get_actor()

        # find the qvalue for each action
        action_to_expected_qvalue = dict()
        for action, next_state in node.action_to_next_state.items():

            next_node = self.get_node(next_state)
            if next_node is None:
                # TODO: this is a hack, fix it
                next_node = self.add_state(next_state)

            # TODO: we forgot to check if the state is terminal here. if terminal, may not have value estimates
            # get the value of the next state using utility estimator
            next_value = self.get_estimated_value(next_node , actor)
            # get the intermediate reward
            reward = node.action_to_actor_to_reward[action][actor]
            # add the value to the expected qvalue
            action_to_expected_qvalue[action] = next_value + reward
        return action_to_expected_qvalue
            
    def get_adjusted_qvalues(self, state: Hashable, ) -> dict[Any, float]:
        '''
        Gets the adjusted qvalues for a given actor.
        Usually the PUCT value is used
        NOTE: this only gets the qvalues for visited actions!!!

        Args:
            node: node to get the adjusted qvalues for
            actor: actor to get the adjusted qvalues for

        Returns:
            action_to_value: dictionary of adjusted qvalues
        '''
        node = self.get_node(state)
        if node is None:
            raise ValueError('Node not found in graph')
        action_to_prob = node.get_action_to_probs()
        # get the expected qvalue for each action
        action_to_expected_qvalue = self.get_qvalues(state)

        # for each action, use the adjuster to get the adjusted qvalue
        action_to_value = dict()
        for action in node.get_actions():
            action_to_value[action] = self.adjuster.adjust(action_to_expected_qvalue[action], action_to_prob[action], node.visits, node.action_to_visits[action])
        return action_to_value
    
    def get_best_action(self, state: Hashable, actions: Optional[set[Hashable]] = None):
        '''
        Gets the best action for the given state
        '''
        adj_qvalues = self.get_adjusted_qvalues(state)
        return max(adj_qvalues, key=adj_qvalues.get)
    
    def get_boltzmann_policy(self, node: ValueNode, temperature: float, adjusted_q=False) -> dict:
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
    
    def get_scheduled_boltzmann_policy(self, node: ValueNode, actor, adjusted_q = False) -> dict:
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

    def get_estimated_value(self, node: ValueNode, actor) -> float:
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

    

    def backpropagate(self, state: Hashable, action, next_state: Hashable):
        '''
        Backpropagates the value from the next state to the current state for each actor

        Args:
            node: node to backpropagate to
            action: action taken to get to the next state
            next_state: next state to backpropagate from
        '''
        node = self.get_node(state)
        if node is None:
            raise ValueError('Node not found in graph')

        # we need to backpropagate for each player in self.players
        for actor in self.players:
            next_node = self.get_node(next_state)

            if next_node is None:
                next_node = self.add_state(next_state)
                # TODO: this is a hack, fix it. new node values will default to 0

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
        node.action_to_visits[action] += 1

    def backpropagate_trajectory(self, trajectory: List[Tuple[Optional[Hashable], Hashable]]):
        '''
        Backpropagates the value from the trajectory to the root.

        Args:
            trajectory: trajectory to backpropagate, list of (action, state) pairs, starting with (None, initial_state).
                        The function processes all states in the trajectory uniformly, including the terminal state.
        '''

        # Start from the last tuple and process backwards to include the terminal state
        for i in range(len(trajectory) - 1, 0, -1):
            _, state = trajectory[i-1]
            action, next_state = trajectory[i]
            self.backpropagate(state, action, next_state)

    def get_backtrack_path(self, state: Hashable) -> List[Tuple[Hashable, Optional[Tuple[Tuple[Any, Any], ...]]]]:
        '''
        Gets the path from the given state to the root of the graph. If a state has multiple parents, it will choose one arbitrarily.

        Args:
            state: The state from which to begin backtracking.

        Returns:
            List of tuples, where each tuple contains a state and the joint action taken to reach that state from its parent,
            or None if the state is the root or has no parent.
        '''
        path = []
        node = self.get_node(state)

        # Traverse backwards from the node to the root.
        while node and node.parents:
            # Select one parent arbitrarily; could be modified to select based on some criteria.
            parent_node = next(iter(node.parents))  # Selects the first parent in the set.

            # Find the action that leads from the parent to the current node.
            for action, next_state in parent_node.action_to_next_state.items():
                if next_state == node.state:
                    path.append((parent_node.state, action))
                    break

            # Move to the parent node.
            node = parent_node

        # Append the initial node or root state if necessary.
        if node:  # This check is to include the root or starting state if applicable.
            path.append((node.state, None))

        # The path is constructed from leaf to root, so we reverse it.
        path.reverse()
        return path

    def add_child_to_state(self, state: Hashable, child_state: Hashable, action: Hashable, rewards: dict[int, float],):
        '''
        Adds a child to a state with the given action and reward

        Args:
            state: state to add the child to
            child_state: state of the child
            action: action taken to get to the child
            reward: reward received for taking the action for each actor
        '''
        node = self.get_node(state)
        if node is None:
            raise ValueError('Node not found in graph')
        child_node = self.get_node(child_state)
        if child_node is None:
            child_node = self.add_state(child_state)
        node.children.add(child_node)
        child_node.parents.add(node)
        node.action_to_next_state[action] = child_state
        node.action_to_actor_to_reward[action] = rewards
        # node.action_to_prob_weights[action] = prob_weight

    def overwrite_state(self, state: Hashable, actor: Optional[int], actions: set[Hashable], actor_to_value_estimates: dict[int, float], notes: dict, prob_weights: Optional[dict[Hashable, float]] = None):
        '''
        Conducts initial inference on a state

        Args:
            state: state to overwrite
            actor: new actor for the state
            actions: new actions for the state
            prob_weights: new prob weights for the state
            actor_to_value_estimates: value estimates for each actor for the state
            notes: new notes for the state
        '''
        node = self.get_node(state)
        if node is None: # node does not exist, create it. this should never happen unless root node
            node = self.add_state(state)

        # overwrite the node
        node.actor = actor   
        node.actions = actions
        if prob_weights is not None:
            node.action_to_prob_weights = prob_weights
        # overwrite the value estimates by setting the list to the [new value]
        for actor, value in actor_to_value_estimates.items():
            node.actor_to_value_estimates[actor] = [value]
        node.notes = notes
        node.is_expanded = True

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

END_STATE = StateTemplate('END_STATE')
    
class ValueGraph2(Graph):
    '''
    A DAG where each node represents a state and each edge represents an action

    Updated version of ValueGraph

    TODO: fix graph so that value estimates contains values for each actor
    '''
    id_to_node: dict[Any, ValueNode2]

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
    
    def add_state(self, state, parent_states=[], child_states=[]):
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
        
    def add_edge(self, parent, child,) -> None:
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
        
    def simulate_trajectory(self, state, num_steps: int = -1) -> List[Tuple[Any, Tuple[Tuple[Any, Any], ...]]]:
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
        # self.logger.info(f'Selected joint action {joint_action} for state {node.state}')
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

    

    def backpropagate(self, node: ValueNode2, action, next_state):
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

    def get_backtrack_path(self, state) -> List[Tuple[Any, Optional[Tuple[Tuple[Any, Any], ...]]]]:
        '''
        Gets the path from the given state to the root of the graph. If a state has multiple parents, it will choose one arbitrarily.

        Args:
            state: The state from which to begin backtracking.

        Returns:
            List of tuples, where each tuple contains a state and the joint action taken to reach that state from its parent,
            or None if the state is the root or has no parent.
        '''
        path = []
        node = self.get_node(state)

        # Traverse backwards from the node to the root.
        while node and node.parents:
            # Select one parent arbitrarily; could be modified to select based on some criteria.
            parent_node = next(iter(node.parents))  # Selects the first parent in the set.

            # Find the action that leads from the parent to the current node.
            for action, next_state in parent_node.action_to_next_state.items():
                if next_state == node.state:
                    path.append((parent_node.state, action))
                    break

            # Move to the parent node.
            node = parent_node

        # Append the initial node or root state if necessary.
        if node:  # This check is to include the root or starting state if applicable.
            path.append((node.state, None))

        # The path is constructed from leaf to root, so we reverse it.
        path.reverse()
        return path


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
    
class PartialValueGraph(ValueGraph):
    '''
    A graph that also keeps track of information sets
    Each information set contains a set of nodes (states) that are indistinguishable to the player
    Whenever we take qvalues, we take the (weighted?) average of the qvalues of the nodes in the information set since the player cannot distinguish between them
    We can estimate the frequency of each node in the information set by the number of visits to the information set (MC estimate)
    '''

    information_set_to_node: dict[Hashable, InformationSetNode] 
    # player_and_node_to_information_set: dict[Tuple[Any, ValueNode], Any]

    def __init__(self, players, adjuster: Union[QValueAdjuster, str] = "no adjust", utility_estimator: Union[UtilityEstimator, str] = "mean", rng = np.random.default_rng()):
        super().__init__(players=players, adjuster=adjuster, utility_estimator=utility_estimator, rng=rng)
        self.information_set_to_node = dict()
        # self.player_and_node_to_information_set = dict()
        self.logger.info(f'PartialValueGraph initialized with players {players}')

    def get_information_set_node(self, information_set: Hashable) -> Optional[InformationSetNode]:
        '''
        Gets the information set node corresponding to the information set. Returns None if the information set does not exist
        '''
        if information_set not in self.information_set_to_node:
            print('Information set not found in graph')
            print('Information set:', information_set)
            print(self.information_set_to_node)
            return None
        else:
            print('Information set found in graph')
        return self.information_set_to_node[information_set]

    def get_information_set(self, node: ValueNode) -> Any:
        '''
        Gets the information set of a state for a given actor
        '''
        # return self.player_and_node_to_information_set[(actor, node)]
        return node.get_acting_player_information_set()
    
    def add_information_set(self, state: Hashable, information_set: Hashable, acting_player: int):
        '''
        Adds adds the state to the information set
        '''
        node = self.get_node(state)
        if node is None:
            raise ValueError('Node not found in graph')
        if information_set not in self.information_set_to_node:
            information_set_node = InformationSetNode(information_set, acting_player)
            self.information_set_to_node[information_set] = information_set_node
        else:
            information_set_node = self.information_set_to_node[information_set]
        information_set_node.add_state(state)
        node.set_acting_player_information_set(information_set)
        node.set_actor(acting_player)
        # add all actions in node to information set node
        information_set_node.add_actions(node.get_actions())

    def get_qvalues_for_information_set(self, information_set: Hashable) -> dict[Hashable, float]:
        '''
        Gets the qvalues for a given actor
        '''
        information_set_node = self.get_information_set_node(information_set)
        if information_set_node is None:
            raise ValueError('Information set node not found in graph')
        acting_player = information_set_node.get_actor()
        action_to_expected_qvalue = dict()
        for action in information_set_node.actions:
            value_estimate_list = []
            for possible_state in information_set_node.get_states_in_set():
                node = self.get_node(possible_state)
                if node is None:
                    raise ValueError('Node not found in graph')
                if action in node.action_to_next_state:
                    next_state = node.action_to_next_state[action]
                    next_node = self.get_node(next_state)
                    if next_node is not None:
                        reward = node.action_to_actor_to_reward[action][acting_player]
                        # extend value_estimate_list with the value estimates of the next node + reward
                        value_estimate_list.extend([value_estimate + reward for value_estimate in next_node.actor_to_value_estimates[acting_player]])
            action_to_expected_qvalue[action] = self.utility_estimator.estimate_list(value_estimate_list)
        print(action_to_expected_qvalue)
        return action_to_expected_qvalue
    
    # def get_adjusted_qvalues_for_information_set(self, information_set: Hashable) -> dict[Hashable, float]:

    #     pass


    def add_child_to_state(self, state: Hashable, child_state: Hashable, action: Any, rewards: dict[int, float],):
        super().add_child_to_state(state, child_state, action, rewards)
        # also add the action to the information set for state
        node = self.get_node(state)
        if node is None:
            raise ValueError('Node not found in graph')
        information_set = self.get_information_set(node)
        information_set_node = self.information_set_to_node[information_set]
        information_set_node.add_actions({action})

    def overwrite_state(self, state: Hashable, actor: Optional[int], actions: set[Hashable], actor_to_value_estimates: dict[int, float], notes: dict, prob_weights: Optional[dict[Hashable, float]] = None, information_set: Hashable = None):
        super().overwrite_state(state, actor, actions, actor_to_value_estimates, notes, prob_weights)

        node = self.get_node(state)
        if node is None:
            raise ValueError('Node not found in graph')
        if actor is not None:
            if information_set is not None: # this option also overwrites the information set for the state
                # add information set node if it does not exist
                print("Adding information set!!!!!!!!!!!!!!!!!!")
                self.add_information_set(state, information_set, actor)
                information_set_node = self.get_information_set_node(information_set)
                # if information_set_node is None:
                #     information_set_node = InformationSetNode(information_set, actor)
                #     self.information_set_to_node[information_set] = information_set_node
                # information_set_node.add_state(state)
                # node.set_acting_player_information_set(information_set)
            else:
                information_set = self.get_information_set(node)
                information_set_node = self.information_set_to_node[information_set]
            information_set_node.add_actions(actions)
        else:
            print("Actor is None??!?!?!?!?!?")

    # def simulate_trajectory_from_information_set(self, information_set: Hashable, information_prior: InformationPrior, num_steps: int = -1, stop_condition: Callable[[ValueNode], bool] | str = lambda x : False) -> list[Tuple[Hashable, Hashable]]:
    #     '''
    #     Since state might be an information set here, we need to sample a state from the information set first before calling super().simulate_trajectory
    #     '''

    #     if information_set in self.information_set_to_node: # choose according to empirical MCTS distribution
    #         information_set_node = self.information_set_to_node[information_set]
    #         state = self.rng.choice(list(information_set_node.get_states_in_set()))
    #     else: 
    #         # we have never simulated this information set before
    #         # in this case, we have two options: (1) simulated starting from the last known full state (like the root) or (2) use some other heuristic to choose a state
    #         # raise ValueError(f"state {state} not found in graph")

    #         # we will use the information prior to select a state
    #         state = information_prior.get_prior_state(information_set)
        
    #     return super().simulate_trajectory(state, num_steps, stop_condition)

    def select_hidden_state(self, information_set: Hashable, information_prior: InformationPrior) -> Hashable:
        '''
        Samples a state from the information set according to the information prior and empirical MCTS distribution over the hidden states in the information set
        '''
        state = information_prior.get_prior_state(information_set)
        return state

    def get_qvalues(self, state: Hashable) -> dict[Hashable, float]:
        '''
        Gets the qvalues for the acting actor in the node
        '''
        node = self.get_node(state)
        if node is None:
            raise ValueError('Node not found in graph')
        information_set = self.get_information_set(node)
        return self.get_qvalues_for_information_set(information_set)
    
    def get_best_action_from_information_set(self, information_set: Hashable, actions: set[Hashable]):
        '''
        Gets the best action for the given state
        '''
        information_set_node = self.get_information_set_node(information_set)
        if information_set_node is None:
            # select random action since we don't have enough information
            return self.rng.choice(list(actions))
        else:
            q_values = self.get_qvalues_for_information_set(information_set)
            return max(q_values, key=q_values.get)
        
class PartialValueGraph2(ValueGraph2):
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
                # get the next state if the action is explored in node1
                if joint_action in node1.action_to_next_state:
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
    
    def get_estimated_value(self, node: ValueNode2, actor) -> float:
        '''
        Gets the estimated value of a node for a given actor

        Args:
            node: node to get the estimated value of
            actor: actor to get the estimated value for

        Returns:
            value: estimated value of the node for the actor
        '''
        
        # since this is a partial value graph, we need to get the value of the information set
        information_set = self.get_information_set(node, actor)
        nodes = self.player_to_information_set_to_set_of_nodes[actor][information_set]
        value_list = []
        for node1 in nodes:
            if not node1.is_done():
                value_list.extend(node1.actor_to_value_estimates[actor])
        return self.utility_estimator.estimate_list(value_list)