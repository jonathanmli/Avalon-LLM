from search_src.searchlightimprove.headers import *
from search_src.searchlightimprove.llm_utils.llm_api_models import GPT35Multi
from search_src.searchlightimprove.prompts.improvement_prompts import IMPROVEMENT_PROMPTS
from search_src.GOPS.baseline_models_GOPS import *
from search_src.GOPS.value_heuristic_evaluators import GOPSValueHeuristicsSSGEvaluator
from search_src.searchlightimprove.analyzers import HeuristicsAnalyzer
from search_src.searchlight.gameplay.simulators import GameSimulator
from search_src.GOPS.examples.abstract_list3 import abstract_list
from search_src.GOPS.examples.func_list3 import func_list
from search_src.utils import setup_logging_environment
from search_src.searchlightimprove.evolvers import ImprovementLibraryEvolver, BeamEvolver, ThoughtBeamEvolver
from search_src.searchlightimprove.prompts.prompt_generators import PromptGenerator
from search_src.searchlightimprove.prompts.improvement_prompts import GOPS_RULES, GOPS_FUNCTION_SIGNATURE
from search_src.Avalon.baseline_models_Avalon import *
from search_src.searchlight.datastructures.graphs import ValueGraph2
from search_src.Avalon.examples.avalon_func import avalon_func_list
# from search_src.Avalon.value_heuristic_evaluators import AvalonValueHeuristicsSSGEvaluator
from search_src.searchlightimprove.rl_evolver import RLValueHeuristicsSSGEvaluator, RLEvolver
import logging
import os
import datetime


import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../configs", config_name="run_modular_improve")
def main(cfg : DictConfig):
    # TODO: see if we can make this more descent
    # create main logger
    logger = logging.getLogger(__name__)
    logger.info('Starting initialization')
    starttime = datetime.datetime.now()

    hydra_working_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.info(f"Outputs for current run will be saved to {hydra_working_dir}")
    # parameters for configuration
    # TODO: move these to a configuration file, use hydra
    
    num_batch_runs = cfg.preset_modular_improve.num_batch_runs
    batch_size = cfg.preset_modular_improve.batch_size
    evolutions = cfg.preset_modular_improve.evolutions
    num_responses = cfg.preset_modular_improve.num_responses
    against_benchmark = cfg.preset_modular_improve.get("against_benchmark", None)
    final_eval_num_batch_runs = cfg.preset_modular_improve.get("final_eval_num_batch_runs", 1)
    save_dir = cfg.get("save_dir", hydra_working_dir)
    num_search_budget = cfg.preset_modular_improve.get("num_search_budget", 16)
    num_random_rollouts = cfg.preset_modular_improve.get("num_random_rollouts", 4)

    logger.info(str(OmegaConf.to_yaml(cfg)))

    # creat rng
    rng = np.random.default_rng(12)

    # configure the environment
    env_name = cfg.env_preset.env_name
    if env_name == 'GOPS':
        # create GOPSValueHeuristicsSSGEvaluator
        GOPS_num_cards = cfg.env_preset.num_cards
        transitor=GOPSForwardTransitor2()
        actor_enumerator=GOPSActorEnumerator()
        action_enumerator=GOPSActionEnumerator()
        start_state=GOPSState2({-1}, tuple(),tuple(), tuple(), GOPS_num_cards)

        # create game simulator
        simulator = GameSimulator(transitor=transitor, 
                                  actor_enumerator=actor_enumerator, action_enumerator=action_enumerator, start_state=start_state,
                                  rng=rng)
        players = {0, 1}

    elif env_name == 'Avalon':
        # get relevant parameters
        num_players = cfg.env_preset.num_players

        # create avalon environment
        env = AvalonGameEnvironment.from_num_players(num_players)
        player_lst = [i for i in range(num_players)]
        player_set = set(player_lst)
        action_enumerator = AvalonActionEnumerator(env)
        transitor = AvalonTransitor(env)
        actor_enumerator = AvalonActorEnumerator()
        start_state = AvalonState.init_from_env(env)
        players = player_set

        # create game simulator
        simulator = GameSimulator(transitor=transitor, 
                                  actor_enumerator=actor_enumerator, action_enumerator=action_enumerator, start_state=start_state, 
                                  rng=rng)

    else:
        raise ValueError(f'Environment {env_name} not supported')
    
    # create evaluator
    evaluator = RLValueHeuristicsSSGEvaluator(simulator=simulator, num_batch_runs=num_batch_runs, players = players, rng=rng, against_benchmark=against_benchmark, search_budget=num_search_budget, random_rollouts=num_random_rollouts, transitor=transitor, actor_enumerator=actor_enumerator, action_enumerator=action_enumerator,)

    # create evolver
    evolver = RLEvolver(evaluator=evaluator, batch_size=batch_size)

    # log how long it took to initialize
    endtime = datetime.datetime.now()
    elapsed_time = endtime - starttime
    logger.info(f'Initialization took {elapsed_time}')

    logger.info('Starting evolution')

    starttime = datetime.datetime.now()

    # evolve 
    evolver.evolve(evolutions)

    # log how long it took to evolve
    endtime = datetime.datetime.now()
    elapsed_time = endtime - starttime
    logger.info(f'Evolution took {elapsed_time}')

    logger.info('Starting final evaluation')

    starttime = datetime.datetime.now()

    # set batch_runs of evaluator to final_eval_num_batch_runs
    evaluator.set_num_batch_runs(final_eval_num_batch_runs)

    endtime = datetime.datetime.now()
    elapsed_time = endtime - starttime
    logger.info(f'Final evaluation took {elapsed_time}')

    evolver.produce_analysis()

    logger.info('Finished')


if __name__ == '__main__':
    setup_logging_environment(log_level=logging.INFO)
    print('Running main')
    main()