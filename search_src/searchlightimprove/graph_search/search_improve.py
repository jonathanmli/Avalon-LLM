from search_src.searchlight.algorithms.best_first_search import BestFirstSearch
from search_src.searchlight.datastructures.graphs import ValueGraph2
from search_src.GOPS.value_heuristic_evaluators import *
from search_src.self_improve.self_improve_search import SelfImprovementInitialInferencer
from search_src.self_improve.llm_api_models import GPT35Multi
from search_src.self_improve.proposers import LLMImprovementProposer
from search_src.self_improve.prompts.improvement_prompts import IMPROVEMENT_PROMPTS
from search_src.searchlight.datastructures.adjusters import QValueAdjuster
from search_src.searchlight.datastructures.estimators import UtilityEstimatorLast
from search_src.GOPS.baseline_models_GOPS import *

def bfs_improve(seed_functions: list) -> ValueGraph2:
    '''
    Use best first search to improve a collection of seed functions

    Args:
        seed_functions: list of seed functions

    Returns:
        graph: graph of improved seed functions
    '''
    # create improvement proposer
    gpt = GPT35Multi(temperature=0.7, num_responses=1)
    proposer = LLMImprovementProposer(gpt, IMPROVEMENT_PROMPTS)

    # create GOPS simulator
    transitor = GOPSForwardTransitor2()
    actor_enumerator = GOPSActorEnumerator()
    action_enumerator = GOPSActionEnumerator()
    simulator = GameSimulator(transitor, actor_enumerator, action_enumerator, GOPS_START_STATE_6)

    # create GOPSValueHeuristicsSSGEvaluator
    evaluator = GOPSValueHeuristicsSSGEvaluator(simulator, num_batch_runs=10)

    # create adjuster
    adjuster = QValueAdjuster()

    # create estimator
    estimator = UtilityEstimatorLast()

    # create graph
    graph = ValueGraph2(adjuster=adjuster, utility_estimator=estimator)

    # create initial inferencer
    initial_inferencer = SelfImprovementInitialInferencer(proposer, evaluator, graph)

    # create best first search algorithm
    search_algorithm = BestFirstSearch(initial_inferencer)

    # add root node
    root_state = State("root", {'score': float('-inf'), 'notes': 'root', 'done': False})
    root_node = graph.add_state(root_state)
    root_node.is_expanded = True

    # add seed functions as children of root node
    for seed_function in seed_functions:
        seed_state = State(seed_function, {'score': float('-inf'), 'notes': 'seed', 'done': False})
        seed_node = graph.add_state(seed_state)
        graph.add_edge(root_state, seed_state)

    # run search
    search_algorithm.expand(graph, root_state)

    return graph

