# from search_src.searchlightimprove.headers import *
from strategist.searchlightimprove.llm_utils.llm_api_models import GPT35Multi
from strategist.searchlightimprove.analyzers import DialogueAnalyzer, LLMCriticAnalyzer, OutcomeDialogueAnalyzer
from strategist.utils import setup_logging_environment
from strategist.searchlightimprove.evolvers import ImprovementLibraryEvolver, BeamEvolver, ThoughtBeamEvolver
from strategist.Avalon.baseline_models_Avalon import *
from strategist.dialogue_improve.prompt_generator import PromptGenerator, SYS_PROMPT
import strategist.searchlightimprove.prompts.prompt_generators as prompt_generators
from strategist.dialogue_improve.prompting_improve import PromptSSGEvaluator, PromptEvaluator
from strategist.dialogue_improve.data_loader import DataLoader
from strategist.Avalon.examples.avalon_func import avalon_best_functions
from strategist.searchlightimprove.value_heuristic_improve import LLMCriticValueHeuristicEvaluator



import logging
import os
import datetime

# TEST_STRATEGIES_MERLIN = [('As Merlin you should not reveal that you know who is Evil.', {'abstract':''}), 
#                           ('As Merlin you should focus on convincing the good players.', {'abstract':''}), 
#                           ('As Merlin you should try to point how the Evil players are acting using evidence.',{'abstract':''})]
# TEST_STRATEGIES_EVIL = [('As Evil you should try to blend in with the good players.',{'abstract':''}), 
#                         ('As Evil you should try to point out the good players that are acting suspiciously.',{'abstract':''}), 
#                         ('As Evil you should try to confuse the good players.',{'abstract':''})]

TEST_STRATEGIES_MERLIN = [("""Questions to think about to play the Merlin role effectively during the discussion phase:

        Q1: Which player seems the most suspicious of you and why?

        Q2: For the player that seems the most suspicious of you, is it worth convincing them that you are on their side? Why or why not?

        Q3: For the player that seems the most suspicious of you, what can you say to convince them that you are on their side?

        Q4: What are some conclusions that you can draw from your answers to the previous questions?

        Example of how to fill out this questionaire:

        Q1: Which player seems the most suspicious of you and why?
        A1: Player 3 seems the most suspicious of me because they have been asking me a lot of questions about my role.

        Q2: For the player that seems the most suspicious of you, is it worth convincing them that you are on their side? Why or why not?
        A2: It is worth convincing Player 3 that I am on their side because they are a key player in the game and I need their support to win.

        Q3: For the player that seems the most suspicious of you, what can you say to convince them that you are on their side?
        A3: I can say that I have never been on a failed quest and that I have been trying to reason with the other players to figure out who the Evil players are.

        Q4: What are some conclusions that you can draw from your answers to the previous questions?
        A4: I should focus on convincing Player 3 that I am on their side and try to get them to support me in the game.""", {'abstract':''}), 
                          ('As Merlin you should focus on convincing the good players.', {'abstract':''}), 
                          ('As Merlin you should try to point how the Evil players are acting using evidence.',{'abstract':''})]

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
    num_fittest_strategies = cfg.preset_modular_improve.get("num_fittest_functions", 1)
    num_ideas_per_iteration = cfg.preset_modular_improve.get("num_ideas_per_iteration", 1)
    generate_new_seed_functions = cfg.preset_modular_improve.get("generate_new_seed_functions", False)
    evolver_name = cfg.evolver_name
    feedback_method = cfg.get("feedback_method", "simulated")
    role_to_eval = cfg.preset_modular_improve.get("role_to_eval", 0)
    num_search_budget = cfg.preset_modular_improve.get("num_search_budget", 8)
    num_scenarios = cfg.preset_modular_improve.get("num_scenarios", 1)

    stochastic_combinations = cfg.preset_modular_improve.get("stochastic_combinations", True)
    search_guided_sampling = cfg.preset_modular_improve.get("search_guided_sampling", True)
    num_feedback_examples = cfg.preset_modular_improve.num_feedback_examples
    data_path = cfg.get("data_path", "data/avalon_ds.json")
    seed_function_file_directory = cfg.preset_modular_improve.get("seed_function_file_directory", "outputs/2024-05-16/18-17-22")
    select_from_last_generation_only = cfg.preset_modular_improve.select_from_last_generation_only
    add_old_functions_to_evaluation = cfg.preset_modular_improve.add_old_functions_to_evaluation
    forward_fill = cfg.preset_modular_improve.get("forward_fill", False)
    evaluate_old_functions = cfg.preset_modular_improve.get("evaluate_old_functions", False)

    model_type = cfg.model.get("model", "gpt-3.5-turbo")
    

    logger.info(str(OmegaConf.to_yaml(cfg)))

    # note that number of simulations will be O(num_responses^3 * num_batch_runs * batch_size^2)

    # create improvement proposer
    llm_model = GPT35Multi(temperature=0.7, num_responses=num_responses, model=model_type)

    # creat rng
    rng = np.random.default_rng(12)

    # if generate_new_seed_functions is False, load seed functions from seed_function_file_directory
    if not generate_new_seed_functions:
        seed_functions = BeamEvolver.load_seed_functions(seed_function_file_directory)
    else:
        seed_functions = None

    # configure the environment
    env_name = cfg.env_preset.env_name
    if env_name == 'Avalon':
        # get relevant parameters
        num_players = cfg.env_preset.num_players

        # create shared value heuristic
        vh_func_str = avalon_best_functions[0]
        value_heuristic = AvalonLLMFunctionalValueHeuristic(vh_func_str)

        # create avalon environment
        config = AvalonBasicConfig.from_num_players(num_players)
        env = AvalonGameEnvironment(config=config)
        player_lst = [i for i in range(num_players)]
        player_set = set(player_lst)
        role_to_eval_str = config.ROLES[role_to_eval]

        # create prompt generator for avalon
        evaluator_prompt_generator = PromptGenerator(config=config, rules=GAME_RULES, sys_prompt=SYS_PROMPT)

        # create evaluator
        evolver_prompt_generator = prompt_generators.StrategyPromptGenerator(environment_rules=GAME_RULES, role_name=role_to_eval_str)

        data_loader = DataLoader()
        data_loader.load_data('src/dialogue_improve/test_data.json')
        # data_loader.load_data(data_path)

        
        # if role_to_eval == 0:
        #     if generate_new_seed_functions:
        #         seed_functions = None
        #     else:
        #         seed_functions = TEST_STRATEGIES_MERLIN
            

        # elif role_to_eval == 7:
        #     if generate_new_seed_functions:
        #         seed_functions = None
        #     else:
        #         seed_functions = TEST_STRATEGIES_EVIL

            # evaluator = PromptSSGEvaluator(players=player_set, role_to_evaluate=role_to_eval, data_loader=data_loader, llm_model=llm_model, prompt_generator=evaluator_prompt_generator, num_scenarios=num_batch_runs, rng=rng, value_heuristic=value_heuristic, env=env, num_total_game_sims=8, num_search_rollouts=num_search_budget)
        # else:
        #     raise ValueError(f'Role {role_to_eval} not supported')

    else:
        raise ValueError(f'Environment {env_name} not supported')
    

    # evaluator = PromptSSGEvaluator(players=player_set, role_to_evaluate=role_to_eval, data_loader=data_loader, llm_model=llm_model, prompt_generator=evaluator_prompt_generator, num_scenarios=num_scenarios, rng=rng, env=env, value_heuristic=value_heuristic, num_total_game_sims=num_batch_runs, num_search_rollouts=num_search_budget)
    final_evaluator = evaluator = PromptEvaluator(players=player_set, role_to_evaluate=role_to_eval, data_loader=data_loader, llm_model=llm_model, prompt_generator=evaluator_prompt_generator, num_batch_runs=num_scenarios, rng=rng)


    if feedback_method == 'simulated':
        evaluator = final_evaluator
        # create analyzer
        analyzer = DialogueAnalyzer(evolver_prompt_generator)
    elif feedback_method == 'critic':
        evaluator = LLMCriticValueHeuristicEvaluator(llm_model=llm_model, prompt_generator=evolver_prompt_generator)
        # create analyzer
        analyzer = LLMCriticAnalyzer()
    elif feedback_method == 'outcome':
        evaluator = final_evaluator
        analyzer = OutcomeDialogueAnalyzer(evolver_prompt_generator)
    else:
        raise ValueError(f'Feedback method {feedback_method} not supported')
        
    
    


    # create evolver
    # if evolver_name == 'ImprovementLibrary':
    #     evolver = ImprovementLibraryEvolver(evaluator=evaluator, model=llm_model, analyzer=analyzer, batch_size=batch_size, seed_functions=seed_functions, prompt_generator = evolver_prompt_generator, num_fittest_functions=num_fittest_strategies, num_ideas_per_iteration=num_ideas_per_iteration, evaluate_old_functions=evaluate_old_functions )
    # elif evolver_name == 'Beam':
    #     evolver = BeamEvolver(evaluator=evaluator, model=llm_model, analyzer=analyzer, seed_functions=seed_functions, prompt_generator=evolver_prompt_generator, batch_size=batch_size, num_fittest_functions=num_fittest_strategies, evaluate_old_functions=evaluate_old_functions )
    # elif evolver_name == 'ThoughtBeam':
    #     evolver = ThoughtBeamEvolver(evaluator=evaluator, model=llm_model, analyzer=analyzer, seed_functions=seed_functions, prompt_generator=evolver_prompt_generator, batch_size=batch_size, num_fittest_functions=num_fittest_strategies, evaluate_old_functions=evaluate_old_functions )
    # else:
    #     raise ValueError(f'Evolver {evolver_name} not supported')
    
    # create evolver
    if evolver_name == 'ImprovementLibrary':
        evolver = ImprovementLibraryEvolver(evaluator=evaluator, model=llm_model, analyzer=analyzer, batch_size=batch_size, seed_functions=seed_functions, prompt_generator = evolver_prompt_generator, num_fittest_functions=num_fittest_strategies, num_ideas_per_iteration=num_ideas_per_iteration, select_from_last_generation_only=False, add_old_functions_to_evaluation=add_old_functions_to_evaluation, forward_fill=forward_fill, evaluate_old_functions=False)
    elif evolver_name == 'Beam':
        evolver = BeamEvolver(evaluator=evaluator, model=llm_model, analyzer=analyzer, batch_size=batch_size, seed_functions=seed_functions, prompt_generator = evolver_prompt_generator, num_fittest_functions=num_fittest_strategies, select_from_last_generation_only=False, add_old_functions_to_evaluation=add_old_functions_to_evaluation, forward_fill=forward_fill, evaluate_old_functions=False)
    elif evolver_name == 'ThoughtBeam':
        evolver = ThoughtBeamEvolver(evaluator=evaluator, model=llm_model, analyzer=analyzer, batch_size=batch_size, seed_functions=seed_functions, prompt_generator = evolver_prompt_generator, num_fittest_functions=num_fittest_strategies,  select_from_last_generation_only=False, add_old_functions_to_evaluation=add_old_functions_to_evaluation, forward_fill=forward_fill, evaluate_old_functions=False)
    elif evolver_name == 'Greedy':
        evolver = BeamEvolver(evaluator=evaluator, model=llm_model, analyzer=analyzer, batch_size=batch_size, seed_functions=seed_functions, prompt_generator = evolver_prompt_generator, num_fittest_functions=1, select_from_last_generation_only=True, add_old_functions_to_evaluation=False, forward_fill=forward_fill, evaluate_old_functions=False)
    elif evolver_name == 'Line':
        evolver = BeamEvolver(evaluator=evaluator, model=llm_model, analyzer=analyzer, batch_size=1, seed_functions=seed_functions, prompt_generator = evolver_prompt_generator, num_fittest_functions=1, select_from_last_generation_only=True, add_old_functions_to_evaluation=False, forward_fill=forward_fill, evaluate_old_functions=False)
        evolutions = evolutions * batch_size
        num_batch_runs = num_batch_runs // batch_size
    else:
        raise ValueError(f'Evolver {evolver_name} not supported')
    
    final_evaluator.set_num_batch_runs(num_batch_runs)

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

    # log num_calls and num_generated_responses from LLM model
    logger.info(f'num_calls: {llm_model.num_calls}')
    logger.info(f'expense: ${llm_model.total_expense}')
    logger.info(f'num_generated_responses: {llm_model.num_generated_responses}')

    logger.info('Starting final evaluation')

    starttime = datetime.datetime.now()

    if feedback_method == 'critic':
        # set batch_runs of evaluator to final_eval_num_batch_runs
        # final_evaluator.set_num_batch_runs(num_batch_runs)
        # evaluator.set_filler_heuristic(None)
        evolver.evaluator = final_evaluator

        # produce analysis
        evolver.produce_analysis(save_directory=save_dir, use_estimated_as_final=False)
    else:
        # produce analysis
        evolver.produce_analysis(save_directory=save_dir, use_estimated_as_final=True)

    endtime = datetime.datetime.now()
    elapsed_time = endtime - starttime
    logger.info(f'Final evaluation took {elapsed_time}')

    # produce results
    evolver.produce_figures(save_dir)


if __name__ == '__main__':
    setup_logging_environment(log_level=logging.INFO)
    print('Running main')
    main()