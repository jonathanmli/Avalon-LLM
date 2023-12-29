from beliefs import Graph, MaxValueNode, MinValueNode, RandomValueNode
from headers import *
from collections import deque

class Search:
    '''
    Abstract class for search algorithms
    '''
    def __init__(self, graph: Graph, forward_predictor, forward_enumerator, value_heuristic):
        self.graph = graph
        self.forward_predictor = forward_predictor
        self.forward_enumerator = forward_enumerator
        self.value_heuristic = value_heuristic

    def expand(self, node_id):
        '''
        Expand starting from a node
        '''
        raise NotImplementedError
    
class BFS(Search):
    '''
    Used to perform breadth-first search
    '''
    def __init__(self, graph: Graph, forward_predictor, forward_enumerator, value_heuristic):
        super().__init__(graph, forward_predictor, forward_enumerator, value_heuristic)
        # self.queue = deque()
        # self.visited = set()
        # self.queue.append(graph.root)
        # self.visited.add(graph.root.id)

    def expand(self, node_id, depth=3, revise=False):
        '''
        Expand starting from a node
        
        Args:
            node_id: id of the node to expand
            depth: depth to expand to
            revise: whether to revise the graph or not
        '''
        queue = deque()
        node = self.graph.nodes[node_id]
        queue.append(node)

        while queue:
            node = queue.popleft()
            if node.depth < depth:
                # add children to queue
                for action in self.forward_enumerator.enumerate(node.state):
                    next_state = self.forward_predictor.predict(node.state, action)
                    child = Node(next_state, node, action)
                    if child.id not in self.graph.nodes:
                        self.graph.add_node(child)
                        queue.append(child)
                    else:
                        child = self.graph.nodes[child.id]
                        child.parents.append(node)
                        node.children.append(child)
        return self.graph
 