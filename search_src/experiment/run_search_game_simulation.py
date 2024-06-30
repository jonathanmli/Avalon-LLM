from search_src.experiment.compare_search import *
from search_src.GOPS.baseline_models_GOPS import *
from search_src.GOPS.engine import *
from search_src.searchlight.datastructures.graphs import ValueGraph2, PartialValueGraph
import numpy as np
from search_src.searchlight.headers import *
from search_src.searchlight.datastructures.adjusters import *
from search_src.searchlight.datastructures.beliefs import *
from search_src.searchlight.datastructures.estimators import *
from search_src.searchlight.classic_models import *
from tqdm import tqdm
from search_src.searchlight.algorithms.mcts_search import SMMonteCarlo
# from search_src.searchlightimprove.llm_utils.llm_api_models import GPT35
from search_src.GOPS.examples.func_list import func_list, func1, test_func
from search_src.Avalon.examples.avalon_func import avalon_func_list
from search_src.searchlight.algorithms.full_search import SMMinimax
from search_src.searchlight.gameplay.simulators import *
####
from search_src.Avalon.baseline_models_Avalon import *
from search_src.utils import setup_logging_environment
from types import MappingProxyType


import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import datetime

# from .avalon_func import func_list

@hydra.main(version_base=None, config_path="../configs", config_name="run_search_game_simulation")
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info('Starting initialization')
    starttime = datetime.datetime.now()

    # create rng
    rng = np.random.default_rng(12)
    ###############

    # get config
    game_name = 'Avalon'
    num_players = cfg.preset_search_game_simulation.num_players
    num_games = cfg.preset_search_game_simulation.num_games
    num_random_rollouts = cfg.preset_search_game_simulation.num_random_rollouts
    search_node_budget = cfg.preset_search_game_simulation.search_node_budget
    search_num_rollouts = cfg.preset_search_game_simulation.search_num_rollouts

    env = AvalonGameEnvironment.from_num_players(num_players)
    player_lst = [i for i in range(num_players)]
    player_set = set(player_lst)

    # create config
    if game_name == 'GOPS':
        action_enumerator = GOPSActionEnumerator()
        forward_transitor = GOPSForwardTransitor2()
        actor_enumerator = GOPSActorEnumerator()
        start_state = GOPS_START_STATE_6
        players = {0, 1}
    elif game_name == 'Avalon':
        action_enumerator = AvalonActionEnumerator(env)
        forward_transitor = AvalonTransitor(env)
        actor_enumerator = AvalonActorEnumerator()
        start_state = AvalonState.init_from_env(env)
        players = player_set
    else:
        raise ValueError('Invalid game name')

    # create q adjuster and utility estimator
    q_adjuster = PUCTAdjuster()
    utility_estimator = UtilityEstimatorMean()
    policy_predictor = PolicyPredictor()

    # create graph, one for each player
    graphs = dict()
    for player in players:
        graphs[player] = PartialValueGraph(adjuster=q_adjuster, utility_estimator=utility_estimator, rng=rng, players=players)

    value_heuristics = dict()
    inferencers = dict()
    # create a value heuristic (and inferencer) for each player
    for player in players:
        value_heuristics[player] = RandomRolloutValueHeuristic(actor_enumerator, action_enumerator,forward_transitor, players=players, num_rollouts=num_random_rollouts, rng=rng)
        inferencers[player] = PackageInitialInferencer(forward_transitor, action_enumerator, 
                                                 policy_predictor, actor_enumerator, 
                                                 value_heuristics[player])
        
    # set player 0 to use LLMFunctionalValueHeuristic
    value_heuristics[0] = AvalonLLMFunctionalValueHeuristic(avalon_func_list[0])

    
    # create searches, one for each player
    searches = dict()
    for player in players:
        searches[player] = SMMonteCarlo(initial_inferencer=inferencers[player], rng=rng, node_budget=search_node_budget, num_rollout=search_num_rollouts)
    
    # create game simulator
    # simulator = GameSimulator(forward_transitor, actor_enumerator, action_enumerator, start_state)
    simulator = GameSimulator(forward_transitor, actor_enumerator, action_enumerator, start_state, rng = rng)

    # create agents, one for each player
    agents = dict()
    for player in players:
        # create a search agent
        agents[player] = SearchAgent(search=searches[player], graph=graphs[player], rng=rng, player=player)
    # add a random agent for actor -1
    agents[-1]= RandomAgent(rng)

    # freeze agents
    agents = MappingProxyType(agents)
    # agents = tuple(agents.values())

    # end initialization
    endtime = datetime.datetime.now()
    logger.info(f'Initialization took {endtime - starttime}')
    starttime = datetime.datetime.now()

    # print agents keys
    logger.debug(f'Agents keys 1: {agents.keys()}')

    # play games
    avg_scores, trajectories = simulator.simulate_games(agents=agents,num_games=num_games, start_state=start_state, display=True, random_initialize_start=True)

    # end simulation
    endtime = datetime.datetime.now()
    logger.info(f'Simulation took {endtime - starttime}')
    starttime = datetime.datetime.now()

    # log results
    logger.info(f'Average scores: {avg_scores}')
    logger.info(f'Trajectories: {trajectories}')
    # log score for each player
    for player in players:
        logger.info(f'Player {player} score: {avg_scores[player]}')

if __name__ == "__main__":
    setup_logging_environment(log_level=logging.INFO)
    main()