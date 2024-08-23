from strategist.searchlightimprove.headers import *
from strategist.searchlightimprove.llm_utils.llm_api_models import GPT35Multi
from strategist.GOPS.baseline_models_GOPS import *
# from search_src.GOPS.value_heuristic_evaluators import GOPSValueHeuristicsSSGEvaluator
from strategist.searchlightimprove.analyzers import HeuristicsAnalyzer
from strategist.searchlight.gameplay.simulators import GameSimulator
from strategist.GOPS.examples.abstract_list3 import abstract_list
from strategist.GOPS.examples.func_list3 import func_list
from strategist.utils import setup_logging_environment
from strategist.searchlightimprove.evolvers import ImprovementLibraryEvolver, BeamEvolver, ThoughtBeamEvolver
from strategist.searchlightimprove.prompts.prompt_generators import PromptGenerator
from strategist.Avalon.baseline_models_Avalon import *
from strategist.searchlight.datastructures.graphs import ValueGraph
from strategist.Avalon.examples.avalon_func import avalon_func_list
from strategist.Avalon.examples.avalon_abstract import avalon_abstract_list
# from search_src.Avalon.value_heuristic_evaluators import AvalonValueHeuristicsSSGEvaluator
from strategist.searchlightimprove.value_heuristic_improve import ValueHeuristicsSSGEvaluator
from strategist.GOPS.prompt_generator import GOPSValuePromptGenerator
from strategist.Avalon.prompt_generator import AvalonValuePromptGenerator
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
    num_fittest_functions = cfg.preset_modular_improve.get("num_fittest_functions", 1)
    num_ideas_per_iteration = cfg.preset_modular_improve.get("num_ideas_per_iteration", 1)
    num_search_budget = cfg.preset_modular_improve.get("num_search_budget", 16)
    num_random_rollouts = cfg.preset_modular_improve.get("num_random_rollouts", 4)
    generate_new_seed_functions = cfg.preset_modular_improve.get("generate_new_seed_functions", False)
    evolver_name = cfg.evolver_name
    stochastic_combinations = cfg.preset_modular_improve.get("stochastic_combinations", True)
    search_guided_sampling = cfg.preset_modular_improve.get("search_guided_sampling", True)
    num_feedback_examples = cfg.preset_modular_improve.get("num_feedback_examples", 3)

    model_type = cfg.model.get("model", "gpt-3.5-turbo")

    logger.info(str(OmegaConf.to_yaml(cfg)))

    # note that number of simulations will be O(num_responses^3 * num_batch_runs * batch_size^2)

    # create improvement proposer
    gpt = GPT35Multi(temperature=0.7, num_responses=num_responses, model=model_type)

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
        players = {0, 1}

        # create prompt generator
        prompt_generator = GOPSValuePromptGenerator()

        if generate_new_seed_functions:
            seed_functions = None
        else:
            seed_functions = [(func, {'abstract': abstract}) for func, abstract in zip(func_list, abstract_list)]
        # create game simulator
        simulator = GameSimulator(transitor=transitor, 
                                  actor_enumerator=actor_enumerator, action_enumerator=action_enumerator, start_state=start_state,
                                  rng=rng)
        # create evaluator
        # evaluator = GOPSValueHeuristicsSSGEvaluator(simulator=simulator, num_batch_runs=num_batch_runs, players = {0,1}, against_benchmark=against_benchmark)

        # create check_function
        check_function = LLMFunctionalValueHeuristic.test_evaluate_static
        parse_function = LLMFunctionalValueHeuristic.parse_llm_function

        partial_information = False

        llm_func_value_heuristic_class = LLMFunctionalValueHeuristic

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
        start_state = AvalonState.start_state_from_config(env.config)
        players = player_set

        # create prompt generator for avalon
        prompt_generator = AvalonValuePromptGenerator()

        if generate_new_seed_functions:
            seed_functions = None
        else:
            assert len(avalon_func_list) == len(avalon_abstract_list)
            seed_functions = [(func, {'abstract': abstract}) for func, abstract in zip(avalon_func_list, avalon_abstract_list)]

        # create game simulator
        simulator = GameSimulator(transitor=transitor, 
                                  actor_enumerator=actor_enumerator, action_enumerator=action_enumerator, start_state=start_state, 
                                  rng=rng)

        check_function = AvalonLLMFunctionalValueHeuristic.test_evaluate_static
        parse_function = AvalonLLMFunctionalValueHeuristic.parse_llm_generated_function

        partial_information = True
        llm_func_value_heuristic_class = AvalonLLMFunctionalValueHeuristic
    else:
        raise ValueError(f'Environment {env_name} not supported')
    
    # create evaluator
    evaluator = ValueHeuristicsSSGEvaluator(
        simulator=simulator,
        transitor=transitor,
        actor_enumerator=actor_enumerator,
        action_enumerator=action_enumerator,
        check_function=check_function,
        llm_func_value_heuristic_class=llm_func_value_heuristic_class,
        num_batch_runs=num_batch_runs,
        players=players,
        rng=rng,
        against_benchmark=against_benchmark,
        search_budget=num_search_budget,
        random_rollouts=num_random_rollouts, 
        partial_information=partial_information,
        stochastic_combinations=stochastic_combinations,
    )

    evaluator.set_num_batch_runs(num_batch_runs)

    # create analyzer
    analyzer = HeuristicsAnalyzer(num_samples=num_feedback_examples, prompt_generator=prompt_generator, search_guided_sampling=search_guided_sampling)

    # create evolver
    if evolver_name == 'ImprovementLibrary':
        evolver = ImprovementLibraryEvolver(evaluator=evaluator, model=gpt, analyzer=analyzer, batch_size=batch_size, seed_functions=seed_functions, check_function=check_function, prompt_generator = prompt_generator, parse_function=parse_function, num_fittest_functions=num_fittest_functions, num_ideas_per_iteration=num_ideas_per_iteration)
    elif evolver_name == 'Beam':
        evolver = BeamEvolver(evaluator=evaluator, model=gpt, analyzer=analyzer, seed_functions=seed_functions, prompt_generator=prompt_generator, check_function=check_function, parse_function=parse_function, batch_size=batch_size, num_fittest_functions=num_fittest_functions)
    elif evolver_name == 'ThoughtBeam':
        evolver = ThoughtBeamEvolver(evaluator=evaluator, model=gpt, analyzer=analyzer, seed_functions=seed_functions, prompt_generator=prompt_generator, check_function=check_function, parse_function=parse_function, batch_size=batch_size, num_fittest_functions=num_fittest_functions)
    else:
        raise ValueError(f'Evolver {evolver_name} not supported')

    # log how long it took to initialize
    endtime = datetime.datetime.now()
    elapsed_time = endtime - starttime
    logger.info(f'Initialization took {elapsed_time}')

    logger.info('Starting generation')

    starttime = datetime.datetime.now()

    # generate
    evolver.generate_and_save_seed_functions(save_directory=save_dir)

    # log how long it took to evolve
    endtime = datetime.datetime.now()
    elapsed_time = endtime - starttime
    logger.info(f'Generation took {elapsed_time}')

    # log num_calls and num_generated_responses from LLM model
    logger.info(f'num_calls: {gpt.num_calls}')
    logger.info(f'expense: ${gpt.total_expense}')
    logger.info(f'num_generated_responses: {gpt.num_generated_responses}')

    # make sure we can load the seed functions
    seed_functions = evolver.load_seed_functions(save_directory=save_dir)

    # log the seed functions
    for func, info in seed_functions:
        logger.info(f'Function: {func}')
        logger.info(f'Info: {info}')



if __name__ == '__main__':
    setup_logging_environment(log_level=logging.INFO)
    print('Running main')
    main()