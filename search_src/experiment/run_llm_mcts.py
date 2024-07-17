from search_src.experiment.compare_search import *
from search_src.GOPS.baseline_models_GOPS import *
from search_src.GOPS.engine import *
from search_src.searchlight.datastructures.graphs import ValueGraph
import numpy as np
from search_src.searchlight.headers import *
from search_src.searchlight.datastructures.adjusters import *
from search_src.searchlight.datastructures.beliefs import *
from search_src.searchlight.datastructures.estimators import *
from search_src.searchlight.classic_models import *
from tqdm import tqdm
from search_src.searchlight.algorithms.mcts_search import SMMonteCarlo
from search_src.GOPS.examples.func_list import func_list, func1, test_func
from search_src.searchlight.algorithms.full_search import SMMinimax
from search_src.searchlight.gameplay.simulators import *

def main():
    # TODO: make this generalizable for Avalon? no create separate file is better

    # create random state
    random_state = np.random.RandomState(12)
    # create rng
    rng = np.random.default_rng(12)

    # create q adjuster and utility estimator
    q_adjuster = PUCTAdjuster()
    utility_estimator = UtilityEstimatorLast()

    # create graph, one for each player
    graph_1 = ValueGraph(adjuster=q_adjuster, utility_estimator=utility_estimator, rng=rng, players={0,1})
    graph_2 = ValueGraph(adjuster=q_adjuster, utility_estimator=utility_estimator, rng=rng, players={0,1})

    # create config
    action_enumerator = GOPSActionEnumerator()
    action_predictor = GOPSRandomActionPredictor()
    forward_transitor = GOPSForwardTransitor2()
    actor_enumerator = GOPSActorEnumerator()
    value_heuristic_1 = RandomRolloutValueHeuristic(actor_enumerator, action_enumerator,
                                                    forward_transitor, num_rollouts=10, 
                                                    rng=rng)
    value_heuristic_2 = RandomRolloutValueHeuristic(actor_enumerator, action_enumerator,
                                                    forward_transitor, num_rollouts=10, 
                                                    rng=rng)
    # value_heuristic_1 = LLMFunctionalValueHeuristic(GPT35())
    # value_heuristic_2 = LLMFunctionalValueHeuristic(GPT35())
    # value_heuristic_2 = LLMFunctionalValueHeuristic(GPT35())

    initial_inferencer_1 = GOPSInitialInferencer2(forward_transitor, action_enumerator, 
                                                 PolicyPredictor(), actor_enumerator, 
                                                 value_heuristic_1)

    initial_inferencer_2 = GOPSInitialInferencer2(forward_transitor, action_enumerator,
                                                  PolicyPredictor(), actor_enumerator,
                                                  value_heuristic_2)
    
    # create search
    search_1 = SMMonteCarlo(initial_inferencer=initial_inferencer_1, rng=rng, node_budget=16, num_rollout=32)
    search_2 = SMMonteCarlo(initial_inferencer=initial_inferencer_2, rng=rng, node_budget=16, num_rollout=32)
    # search_2 = SMMinimax(forward_transistor=forward_transitor, value_heuristic=value_heuristic_2,
    #                      actor_enumerator=actor_enumerator, action_enumerator=action_enumerator,
    #                         action_predictor=action_predictor, depth=3, node_budget=16)

    # create game
    # num_cards = 6
    # config = GOPSConfig(num_turns=num_cards, random_state=random_state)
    # env = GOPSEnvironment(config)

    # create game simulator
    simulator = GameSimulator(forward_transitor, actor_enumerator, action_enumerator, GOPS_START_STATE_6)

    num_games = 10

    # return compare_search(search_1, search_2, graph_1, graph_2, num_games, env)
    return compare_search2(search_1, search_2, graph_1, graph_2, simulator, num_games, rng)

if __name__ == "__main__":
    main()