from beliefs import Graph, MaxValueNode, MinValueNode, RandomValueNode, ValueGraph
from headers import *
from collections import deque

class Search:
    '''
    Abstract class for search algorithms
    '''
    def __init__(self, forward_predictor: ForwardPredictor, forward_enumerator: ForwardEnumerator, value_heuristic: ValueHeuristic, action_enumerator: ActionEnumerator):
        # self.graph = graph
        self.forward_predictor = forward_predictor
        self.forward_enumerator = forward_enumerator
        self.value_heuristic = value_heuristic
        self.action_enumerator = action_enumerator

    def expand(self, node_id):
        '''
        Expand starting from a node
        '''
        raise NotImplementedError
    
class ValueBFS(Search):
    '''
    Used to perform breadth-first search
    '''
    def __init__(self, forward_predictor: ForwardPredictor, forward_enumerator: ForwardEnumerator, value_heuristic: ValueHeuristic, action_enumerator: ActionEnumerator):
        super().__init__(forward_predictor, forward_enumerator, value_heuristic, action_enumerator)
        # self.queue = deque()
        # self.visited = set()
        # self.queue.append(graph.root)
        # self.visited.add(graph.root.id)

    def expand(self, graph: ValueGraph, state: State, prev_node = None, depth=3, revise=False):
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
            next_states = set()
            next_state_to_values = dict()

            if state.state_type == state.STATE_TYPES[0]: # max
                actions = self.action_enumerator.enumerate(state)
                for action in actions:
                    next_states.add(self.forward_enumerator.enumerate(state, action))
                    
                for next_state in next_states:
                    value = self.expand(graph, next_state, depth-1, revise)
                    next_state_to_values[next_state] = value

                for action in actions:
                    # calculate expected value
                    expected_value = 0.0
                    for next_state in next_states:
                        expected_value += next_state_to_values[next_state] * self.forward_predictor.predict(state, action, next_state)  
                
            if revise:
                node.value = value
            return value

        # queue = deque()
        # node = self.graph.nodes[node_id]
        # queue.append(node)

        # while queue:
        #     node = queue.popleft()
        #     if node.depth < depth:
        #         # add children to queue
        #         for action in self.forward_enumerator.enumerate(node.state):
        #             next_state = self.forward_predictor.predict(node.state, action)
        #             child = Node(next_state, node, action)
        #             if child.id not in self.graph.nodes:
        #                 self.graph.add_node(child)
        #                 queue.append(child)
        #             else:
        #                 child = self.graph.nodes[child.id]
        #                 child.parents.append(node)
        #                 node.children.append(child)
        # return self.graph
        
 