from strategist.searchlightimprove.headers import *
from strategist.GOPS.baseline_models_GOPS import *
# from search_src.GOPS.value_heuristic_evaluators import GOPSValueHeuristicsSSGEvaluator
from strategist.searchlight.gameplay.simulators import GameSimulator
from strategist.GOPS.examples.abstract_list3 import abstract_list
from strategist.GOPS.examples.func_list3 import func_list
from strategist.Avalon.baseline_models_Avalon import *
from strategist.Avalon.examples.avalon_func import avalon_func_list
from strategist.searchlightimprove.value_heuristic_improve import ValueHeuristicsSSGEvaluator
from strategist.utils import setup_logging_environment

import logging
import os
import datetime

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import plotly.express as px

def retrieve_top_k_functions_from_folder(folder_path: str, k: int) -> list:
    '''
    In the folder, there are N csv files, one for each run which generates some number of functions. Each function is a row in the csv file.
    You are to retrieve the top k functions from each csv file according to the column 'final_score' and combine the into a list of functions. You just need to retrieve the function string from the 'function' column.
    '''
    functions = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder_path, file))
            df = df.sort_values(by='final_score', ascending=False)
            functions.extend(df['function'].values[:k])
    return functions

def retrieve_categorized_functions_from_directory(folder_path: str, k: int) -> dict:
    '''
    In the folder, there are C folders, one for each category.
    You are to, for each folder, use the function retrieve_top_k_functions_from_folder to retrieve the top k functions from each csv file in the folder and combine them into a list of functions.
    The return a dictionary where the key is the category nam (name of folder) and the value is the list of functions for that category.
    '''
    categorized_functions = {}
    for category in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, category)):
            categorized_functions[category] = retrieve_top_k_functions_from_folder(os.path.join(folder_path, category), k)
    return categorized_functions

@hydra.main(version_base=None, config_path="../configs", config_name="run_batch_evaluate")
def main(cfg : DictConfig):
    # create main logger
    logger = logging.getLogger(__name__)

    starttime = datetime.datetime.now()

    hydra_working_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.info(f"Outputs for current run will be saved to {hydra_working_dir}")
    logger.info(str(OmegaConf.to_yaml(cfg)))

    num_batch_runs = cfg.preset_batch_evaluate.num_batch_runs
    save_dir = cfg.get("save_dir", hydra_working_dir)
    against_benchmark = cfg.preset_batch_evaluate.get("against_benchmark", True)
    num_search_budget = cfg.preset_batch_evaluate.get("num_search_budget", 16)
    num_random_rollouts = cfg.preset_batch_evaluate.get("num_random_rollouts", 4)
    evaluating_directory = cfg.preset_batch_evaluate.get("evaluating_directory", None)
    num_top_k_functions_per_category = cfg.preset_batch_evaluate.get("num_top_k_functions_per_category", 3)
    # create rng
    rng = np.random.default_rng(12)

    # configure the environment
    env_name = cfg.env_preset.env_name

    # retrieve the functions from the directory
    categorized_functions = retrieve_categorized_functions_from_directory(evaluating_directory, num_top_k_functions_per_category)

    # expand the dictionary into a list of functions and categories
    functions = []
    function_categories = []
    for category, funcs in categorized_functions.items():
        functions.extend(funcs)
        function_categories.extend([category] * len(funcs))

    if env_name == 'GOPS':
        # create GOPSValueHeuristicsSSGEvaluator
        GOPS_num_cards = cfg.env_preset.num_cards
        transitor=GOPSForwardTransitor2()
        actor_enumerator=GOPSActorEnumerator()
        action_enumerator=GOPSActionEnumerator()
        start_state=GOPSState2({-1}, tuple(),tuple(), tuple(), GOPS_num_cards)
        players = {0, 1}

        # create game simulator
        simulator = GameSimulator(transitor=transitor, actor_enumerator=actor_enumerator, action_enumerator=action_enumerator, start_state=start_state)

        # create evaluator
        # evaluator = GOPSValueHeuristicsSSGEvaluator(simulator=simulator, num_batch_runs=num_batch_runs, players = {0,1}, against_benchmark=False)

        llm_func_value_heuristic_class = LLMFunctionalValueHeuristic
        check_function = LLMFunctionalValueHeuristic.test_evaluate_static
        partial_info = False
        
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
        
        check_function = AvalonLLMFunctionalValueHeuristic.test_evaluate_static

        llm_func_value_heuristic_class = AvalonLLMFunctionalValueHeuristic
        partial_info = True

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
        partial_information=partial_info,
        stochastic_combinations=True,
    )

    # function_scores, function_notes = evaluator.evaluate(functions)
    # log the function scores and notes
    # for func, score, note in zip(functions, function_scores, function_notes):
    #     logger.info(f'Function: {func}, Score: {score}, Note: {note}')

    # evaluate the new functions
    # use the evaluator to evaluate the fittest functions with benchmark
    function_scores, function_notes, benchmark_scores = evaluator.evaluate_with_benchmark(functions)

    
    # log time taken to evaluate functions
    endtime = datetime.datetime.now()
    logger.info(f'Time taken to evaluate functions: {endtime - starttime}')

    # store the results in a list of dictionaries
    results = []
    for func, score, category in zip(functions, function_scores, function_categories):
        # append info dictionary along with the final score and function and estimated score
        to_append = {'function': func, 'score': score, 'category': category}
        results.append(to_append)

    # sort results by final score
    results = sorted(results, key=lambda x: x['score'], reverse=True)


    # log the estimated scores, final score, generation, and iteration for each function
    # for item in results:
    #     logger.info(f'Function: {item["function"]}, Score: {item["score"]}, Category: {item["category"]}')

    # # log the benchmark scores
    # for benchmark_name, benchmark_score in benchmark_scores.items():
    #     logger.info(f'Benchmark {benchmark_name} score: {benchmark_score}')
    
    # convert results to pd.DataFrame, then save to csv
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(save_dir, 'results.csv'), index=False)
    benchmark_filename = os.path.join(save_dir, 'benchmark_scores.csv')
    pd.DataFrame(benchmark_scores.items(), columns=['Benchmark', 'Score']).to_csv(benchmark_filename, index=False)

    # create a plotly express box plot where x is the category and y is the score
    # add a line for each benchmark score
    # include appropriate labels and title
    fig = px.box(results_df, x='category', y='score', title='Function Scores by Category')
    for benchmark_name, benchmark_score in benchmark_scores.items():
        fig.add_hline(y=benchmark_score, line_dash='dash', line_color='red', annotation_text=f'{benchmark_name} benchmark', annotation_position='bottom right')
    fig.update_layout(xaxis_title='Category', yaxis_title='Score')
    fig.write_html(os.path.join(save_dir, 'results.html'))





if __name__ == '__main__':
    setup_logging_environment()

    # Log a test message
    logging.info('This is a test log message.')

    main()