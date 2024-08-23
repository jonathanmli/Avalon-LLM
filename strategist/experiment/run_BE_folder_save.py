from strategist.searchlightimprove.headers import *
from strategist.GOPS.baseline_models_GOPS import *
# from search_src.GOPS.value_heuristic_evaluators import GOPSValueHeuristicsSSGEvaluator
from strategist.searchlight.gameplay.simulators import GameSimulator
from strategist.GOPS.examples.abstract_list3 import abstract_list
from strategist.GOPS.examples.func_list3 import func_list, best_functions
from strategist.Avalon.baseline_models_Avalon import *
from strategist.Avalon.examples.avalon_func import avalon_func_list
from strategist.searchlightimprove.value_heuristic_improve import ValueHeuristicsSSGEvaluator
from strategist.utils import setup_logging_environment
import statsmodels.api as sm

import logging
import os
import datetime

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import plotly.express as px

COLNAMES = ['function', 'num_calls', 'num_output_tokens', 'num_total_tokens', 'iteration', 'generation']

BENCHMARK_FUNCTION = """def evaluate_state(state) -> tuple[tuple[float, float], dict]:
    score_cards = state[0]
    player_0_played_cards = state[1]
    player_1_played_cards = state[2]
    is_turn = state[3]
    player_0_score = state[4]
    player_1_score = state[5]
    score_deck = list(state[6])  # Convert set to list for indexing
    player_0_hand = list(state[7])  # Convert set to list for indexing
    player_1_hand = list(state[8])  # Convert set to list for indexing
    
    def calculate_round_probabilities(player_0_card, player_1_card):
        player_0_wins = 0
        player_1_wins = 0
        
        if player_0_card > player_1_card:
            player_0_wins += 1
        elif player_1_card > player_0_card:
            player_1_wins += 1
        
        return player_0_wins, player_1_wins
    
    def simulate_future_rounds(player_0_hand, player_1_hand, score_deck):
        player_0_expected_score = player_0_score
        player_1_expected_score = player_1_score
        
        for i in range(min(len(player_0_hand), len(score_deck))):
            player_0_card = player_0_hand[i] if i < len(player_0_hand) else 0
            player_1_card = player_1_hand[i] if i < len(player_1_hand) else 0
            
            player_0_wins, player_1_wins = calculate_round_probabilities(player_0_card, player_1_card)
            
            player_0_expected_score += player_0_wins * score_deck[i]
            player_1_expected_score += player_1_wins * score_deck[i]
        
        if not player_0_hand:
            player_0_expected_score += sum(score_deck[:len(player_1_hand)])
        if not player_1_hand:
            player_1_expected_score += sum(score_deck[:len(player_0_hand)])
        
        dynamic_adjustment = sum(score_deck) / (len(player_0_hand) + len(player_1_hand) + 1)
        
        player_0_expected_score += dynamic_adjustment
        player_1_expected_score -= dynamic_adjustment
        
        return player_0_expected_score, player_1_expected_score, dynamic_adjustment
    
    player_0_expected_score, player_1_expected_score, dynamic_adjustment = simulate_future_rounds(player_0_hand, player_1_hand, score_deck)
    
    intermediate_values = {
        'dynamic_adjustment': dynamic_adjustment
    }
    
    return (player_0_expected_score, player_1_expected_score), intermediate_values"""

def retrieve_top_k_items_from_folder(folder_path: str, k: int, colnames: list[str] = COLNAMES) -> pd.DataFrame:
    '''
    In the folder, there are N csv files, one for each run which generates some number of functions. Each row in the csv file contains information about one function.
    You are to retrieve the top k rows from each csv file according to the column 'final_score' and combine the into a single pd.DataFrame. You only need to retrieve the columns specified in colnames.
    '''
    df_list = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder_path, file))
            df = df.sort_values(by='final_score', ascending=False)
            df = df[colnames]
            df_list.append(df.iloc[:k])
    return pd.concat(df_list)

def compile_function_csvs_from_directory(folder_path: str, k: int) -> pd.DataFrame:
    '''
    In the folder, there are C folders, one for each category.
    You are to, for each folder, use the function retrieve_top_k_items_from_folder to retrieve a df of the the top k functions from each csv file in the folder and combine the dfs from all the folders into a single pd.DataFrame.
    You should also add a column 'category' to the df which contains the name of the category (i.e. the name of the folder) for each row.
    '''
    df_list = []
    for category in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, category)):
            df = retrieve_top_k_items_from_folder(os.path.join(folder_path, category), k)
            df['category'] = category
            df_list.append(df)
    return pd.concat(df_list)

def plot_and_save_figures(save_dir:str):
    '''
    Loads the results.csv file from the save_dir and creates the following plots:
    '''

    # load the results.csv file
    function_df = pd.read_csv(os.path.join(save_dir, 'results.csv'))

    # create a plotly express box plot where x is the category and y is the score
    # add a line for each benchmark score
    # include appropriate labels and title
    fig = px.box(function_df, x='category', y='score', title='Function Scores by Category', color='category')
    # for benchmark_name, benchmark_score in benchmark_scores.items():
    #     fig.add_hline(y=benchmark_score, line_dash='dash', line_color='red', annotation_text=f'{benchmark_name} benchmark', annotation_position='bottom right')
    fig.update_layout(xaxis_title='Category', yaxis_title='Score')
    fig.write_html(os.path.join(save_dir, 'results.html'))

    # create a plotly express scatterplot of the final score vs the number of calls, with color as the category
    fig = px.scatter(function_df, x='num_calls', y='score', color='category', title='Function Scores vs Number of Calls')
    fig.update_layout(xaxis_title='Number of Calls', yaxis_title='Score')
    fig.write_html(os.path.join(save_dir, 'calls_vs_score.html'))

    # create a plotly express scatterplot of the final score vs the number of output tokens, with color as the category
    fig = px.scatter(function_df, x='num_output_tokens', y='score', color='category', title='Computation budget vs performance for different methods', trendline="ols")
    fig.update_traces(marker=dict(opacity=0.3))  # Set points opacity to 30%
    fig.update_layout(xaxis_title='Number of Output Tokens', yaxis_title='Score', legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5, title="Method"))
    # fig.update_layout(coloraxis_colorbar=dict(title="Method"))

    # Change plotly theme to 'simple white'
    fig.update_layout(template='simple_white')
    fig.write_html(os.path.join(save_dir, 'output_tokens_vs_score_1.html'))

    

    # put legend on the top
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))

    # # Assuming function_df contains your data
    # fig = px.scatter(function_df, x='num_output_tokens', y='score', color='category', title='Function Scores vs Number of Output Tokens', trendline="ols")
    # fig.update_layout(xaxis_title='Number of Output Tokens', yaxis_title='Score')
    # fig.write_html(os.path.join(save_dir, 'output_tokens_vs_score_1.html'))

    # Assuming function_df contains your data
    fig = px.line(function_df, x='num_output_tokens', y='score', color='category', title='Function Scores vs Number of Output Tokens')
    fig.update_layout(xaxis_title='Number of Output Tokens', yaxis_title='Score')
    fig.write_html(os.path.join(save_dir, 'output_tokens_vs_score_2.html'))

    # Assuming function_df contains your data
    fig = px.scatter(function_df, x='num_output_tokens', y='score', color='category', title='Function Scores vs Number of Output Tokens', trendline="lowess")
    fig.update_layout(xaxis_title='Number of Output Tokens', yaxis_title='Score')
    fig.write_html(os.path.join(save_dir, 'output_tokens_vs_score_3.html'))

    # create a plotly express scatterplot of the final score vs the number of total tokens, with color as the category
    fig = px.scatter(function_df, x='num_total_tokens', y='score', color='category', title='Function Scores vs Number of Total Tokens')
    fig.update_layout(xaxis_title='Number of Total Tokens', yaxis_title='Score')
    fig.write_html(os.path.join(save_dir, 'total_tokens_vs_score.html'))

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
    against_benchmark = cfg.preset_batch_evaluate.get("against_benchmark", False)
    num_search_budget = cfg.preset_batch_evaluate.get("num_search_budget", 16)
    num_random_rollouts = cfg.preset_batch_evaluate.get("num_random_rollouts", 4)
    evaluating_directory = cfg.preset_batch_evaluate.get("evaluating_directory", None)
    num_top_k_functions_per_category = cfg.preset_batch_evaluate.get("num_top_k_functions_per_category", 3)
    # create rng
    rng = np.random.default_rng(12)

    # configure the environment
    env_name = cfg.env_preset.env_name

    # retrieve the function df from the directory
    function_df = compile_function_csvs_from_directory(evaluating_directory, num_top_k_functions_per_category)

    # grab the functions and categories from the df
    functions = function_df['function'].values
    function_categories = function_df['category'].values

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

        gops_best_func = best_functions[0]
        filler_heuristic = LLMFunctionalValueHeuristic(func=gops_best_func)
        
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
        filler_heuristic = None

        
        

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
        use_filler_as_benchmark=True,
        filler_heuristic=filler_heuristic
    )

    evaluator.set_filler_heuristic(filler_heuristic)
    # function_scores, function_notes = evaluator.evaluate(functions)
    # log the function scores and notes
    # for func, score, note in zip(functions, function_scores, function_notes):
    #     logger.info(f'Function: {func}, Score: {score}, Note: {note}')

    # evaluate the new functions
    if not against_benchmark:
        # use the evaluator to evaluate the fittest functions with benchmark
        function_scores, function_notes, benchmark_scores = evaluator.evaluate_with_benchmark(functions)
    else:
        function_scores, function_notes = evaluator.evaluate(functions)
        benchmark_scores = dict()

    
    # log time taken to evaluate functions
    endtime = datetime.datetime.now()
    logger.info(f'Time taken to evaluate functions: {endtime - starttime}')

    # add a new column to the function_df for the final score
    function_df['score'] = function_scores

    # sort function_df by final score
    function_df = function_df.sort_values(by='score', ascending=False)
    
    # save to csv
    function_df.to_csv(os.path.join(save_dir, 'results.csv'), index=False)
    # also save benchmark scores to a csv
    benchmark_df = pd.DataFrame({'benchmark': list(benchmark_scores.keys()), 'score': list(benchmark_scores.values())})
    benchmark_df.to_csv(os.path.join(save_dir, 'benchmark_scores.csv'), index=False)

    plot_and_save_figures(save_dir)







if __name__ == '__main__':
    setup_logging_environment()

    # Log a test message
    logging.info('This is a test log message.')

    main()