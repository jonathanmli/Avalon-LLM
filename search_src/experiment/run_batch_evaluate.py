from search_src.searchlightimprove.headers import *
from search_src.GOPS.baseline_models_GOPS import *
from search_src.GOPS.value_heuristic_evaluators import GOPSValueHeuristicsSSGEvaluator
from search_src.searchlight.gameplay.simulators import GameSimulator
from search_src.GOPS.examples.abstract_list3 import abstract_list
from search_src.GOPS.examples.func_list3 import func_list
from search_src.Avalon.baseline_models_Avalon import *
from search_src.Avalon.examples.avalon_func import avalon_func_list
from search_src.searchlightimprove.value_heuristic_improve import ValueHeuristicsSSGEvaluator
from search_src.utils import setup_logging_environment

import logging
import os
import datetime

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import plotly.express as px

@hydra.main(version_base=None, config_path="../configs", config_name="run_batch_evaluate")
def main(cfg : DictConfig):
    # create main logger
    logger = logging.getLogger(__name__)

    starttime = datetime.datetime.now()

    hydra_working_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.info(f"Outputs for current run will be saved to {hydra_working_dir}")

    num_batch_runs = cfg.preset_batch_evaluate.num_batch_runs
    save_dir = cfg.get("save_dir", hydra_working_dir)
    against_benchmark = cfg.preset_batch_evaluate.get("against_benchmark", True)
    num_search_budget = cfg.preset_batch_evaluate.get("num_search_budget", 16)
    num_random_rollouts = cfg.preset_batch_evaluate.get("num_random_rollouts", 4)

    # create rng
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

        # create game simulator
        simulator = GameSimulator(transitor=transitor, actor_enumerator=actor_enumerator, action_enumerator=action_enumerator, start_state=start_state)

        # create evaluator
        # evaluator = GOPSValueHeuristicsSSGEvaluator(simulator=simulator, num_batch_runs=num_batch_runs, players = {0,1}, against_benchmark=False)

        llm_func_value_heuristic_class = LLMFunctionalValueHeuristic
        check_function = LLMFunctionalValueHeuristic.test_evaluate_static

        # you need to define these yourself
        functions = func_list
        function_categories = [0,0,0,1,1,]
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
        
        # you need to define these yourself
        functions = avalon_func_list
        function_categories = [0,]

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
        random_rollouts=num_random_rollouts
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
    for item in results:
        logger.info(f'Function: {item["function"]}, Score: {item["score"]}, Category: {item["category"]}')

    # log the benchmark scores
    for benchmark_name, benchmark_score in benchmark_scores.items():
        logger.info(f'Benchmark {benchmark_name} score: {benchmark_score}')
    
    # convert results to pd.DataFrame, then save to csv
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(save_dir, 'results.csv'), index=False)

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