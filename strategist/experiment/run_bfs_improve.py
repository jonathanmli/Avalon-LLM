from strategist.searchlightimprove.headers import *
from strategist.searchlightimprove.llm_utils.llm_api_models import GPT35Multi
from strategist.searchlightimprove.proposers import LLMImprovementProposer
from strategist.searchlightimprove.prompts.improvement_prompts import IMPROVEMENT_PROMPTS
from strategist.GOPS.baseline_models_GOPS import *
from strategist.GOPS.value_heuristic_evaluators import GOPSValueHeuristicsSSGEvaluator
from strategist.searchlightimprove.analyzers import HeuristicsAnalyzer
from strategist.searchlight.gameplay.simulators import GameSimulator
from strategist.GOPS.examples.abstract_list3 import abstract_list
from strategist.GOPS.examples.func_list3 import func_list
from strategist.utils import setup_logging_environment

import logging
import os
import datetime

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../configs", config_name="run_modular_improve")
def main(cfg : DictConfig):
    # TODO: see if we can make this more descent
    # if "gops_conf" in cfg.gops_conf:
    #     cfg = cfg.gops_conf
    # create main logger
    logger = logging.getLogger(__name__)
    logger.info('Starting initialization')
    starttime = datetime.datetime.now()

    hydra_working_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.info(f"Outputs for current run will be saved to {hydra_working_dir}")
    # parameters for configuration

    num_batch_runs = cfg.preset_modular_improve.num_batch_runs
    batch_size = cfg.preset_modular_improve.batch_size
    evolutions = cfg.preset_modular_improve.evolutions
    num_responses = cfg.preset_modular_improve.num_responses
    against_benchmark = cfg.preset_modular_improve.get("against_benchmark", None)
    final_eval_num_batch_runs = cfg.preset_modular_improve.get("final_eval_num_batch_runs", 1)
    save_dir = cfg.get("save_dir", hydra_working_dir)
    implement_steps_per_grow = cfg.preset_modular_improve.implement_steps_per_grow

    model_type = cfg.model.get("model", "gpt-3.5-turbo")

    logger.info(str(OmegaConf.to_yaml(cfg)))

    # TODO: move these to a configuration file, use hydra
    # num_batch_runs = cfg.gops_conf.num_batch_runs
    # GOPS_num_cards = cfg.gops_conf.GOPS_num_cards
    # batch_size = cfg.gops_conf.batch_size
    # evolutions = cfg.gops_conf.evolutions
    # num_responses = cfg.gops_conf.num_responses
    # against_benchmark = cfg.gops_conf.get("against_benchmark", None)
    # save_dir = cfg.gops_conf.get("save_dir", hydra_working_dir)
    

    # logger.info(str(OmegaConf.to_yaml(cfg)))

    # note that number of simulations will be O(num_responses^3 * num_batch_runs * batch_size^2)

    # create improvement proposer
    gpt = GPT35Multi(temperature=0.7, num_responses=num_responses, model=model_type)
    proposer = LLMImprovementProposer(gpt, IMPROVEMENT_PROMPTS)

    # # create GOPSValueHeuristicsSSGEvaluator
    # transitor=GOPSForwardTransitor2()
    # actor_enumerator=GOPSActorEnumerator()
    # action_enumerator=GOPSActionEnumerator()
    # start_state=GOPSState2({-1}, tuple(),tuple(), tuple(), GOPS_num_cards)

    env_name = cfg.env_preset.env_name
    if env_name == 'GOPS':
        # create GOPSValueHeuristicsSSGEvaluator
        GOPS_num_cards = cfg.env_preset.num_cards
        transitor=GOPSForwardTransitor2()
        actor_enumerator=GOPSActorEnumerator()
        action_enumerator=GOPSActionEnumerator()
        start_state=GOPSState2({-1}, tuple(),tuple(), tuple(), GOPS_num_cards)
    else:
        raise ValueError(f'Environment {env_name} not supported')

    # create game simulator
    simulator = GameSimulator(transitor=transitor, actor_enumerator=actor_enumerator, action_enumerator=action_enumerator, start_state=start_state)

    # create evaluator
    evaluator = GOPSValueHeuristicsSSGEvaluator(simulator=simulator, num_batch_runs=num_batch_runs, players = {0,1}, against_benchmark=against_benchmark)

    # create analyzer
    analyzer = HeuristicsAnalyzer()

    seed_functions = [(func, abstract) for func, abstract in zip(func_list, abstract_list)]

    # create evolver
    evolver = Evolver(evaluator=evaluator, proposer=proposer, analyzer=analyzer, seed_functions=seed_functions, batch_size=batch_size)

    endtime = datetime.datetime.now()
    elapsed_time = endtime - starttime
    logger.info(f'Initialization took {elapsed_time}')

    logger.info('Starting evolution')

    starttime = datetime.datetime.now()

    # evolve 3 times
    evolver.evolve(evolutions)

    endtime = datetime.datetime.now()
    elapsed_time = endtime - starttime
    logger.info(f'Evolution took {elapsed_time}')

    # log num_calls and num_generated_responses from LLM model
    logger.info(f'num_calls: {gpt.num_calls}')
    logger.info(f'expense: ${gpt.total_expense}')
    logger.info(f'num_generated_responses: {gpt.num_generated_responses}')

    logger.info('Starting final evaluation')
    # set batch_runs of evaluator to 1
    evaluator.set_num_batch_runs(1)

    # produce analysis
    results, benchmark_scores = evolver.produce_analysis()

    # produce results
    evolver.produce_figures(results, benchmark_scores, save_dir)

if __name__ == '__main__':
    setup_logging_environment(log_level=logging.INFO)
    # print('Running main')
    main()