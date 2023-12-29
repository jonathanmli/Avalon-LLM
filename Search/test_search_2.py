from headers import *
from search import *
from beliefs import *
import unittest

class TestSearch(unittest.TestCase):

    # create graph, predictors, and enumerators
    graph = ValueGraph()
    forward_predictor = ForwardPredictor()
    forward_enumerator = ForwardEnumerator()
    value_heuristic = ValueHeuristic()
    action_enumerator = ActionEnumerator()
    random_state_enumerator = RandomStateEnumerator()
    random_state_predictor = RandomStatePredictor()
    opponent_action_enumerator = OpponentActionEnumerator()
    opponent_action_predictor = OpponentActionPredictor()

    # create search
    search = ValueBFS(forward_predictor, forward_enumerator, value_heuristic, action_enumerator, 
                      random_state_enumerator, random_state_predictor,
                      opponent_action_enumerator, opponent_action_predictor)
    
    def test_expand(self):
        # create graph
        graph = self.graph

        # create state
        state = State()

        # expand
        value = self.search.expand(graph, state, depth=3)
        self.assertEqual(value, 0.0)

        # check if node is in graph
        self.assertEqual(graph.get_node(state).value, 0.0)