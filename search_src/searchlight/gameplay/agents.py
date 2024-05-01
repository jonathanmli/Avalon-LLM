from search_src.searchlight.headers import *
from search_src.searchlight.datastructures.graphs import *
# from search_src.Search.adjusters import *
from search_src.searchlight.datastructures.beliefs import *
# from search_src.Search.estimators import *
from search_src.searchlight.algorithms.mcts_search import *

class SearchAgent(Agent):

    def __init__(self, search: Search, graph: ValueGraph2, rng = None, player = -1):
        '''
        Creates a search agent, equipped with a search algorithm and a graph
        '''
        self.search = search
        self.graph = graph
        self.rng = rng
        super().__init__(player)

    def _act(self, state: State, actions):
        # expand the graph first
        self.search.expand(self.graph, state)
        # get the best action from graph
        action = self.search.get_best_action(self.graph, state, self.player) # TODO: temp fix
        return action
    
    def get_graph(self):
        return self.graph
    
 