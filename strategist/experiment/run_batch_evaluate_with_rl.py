from strategist.searchlightimprove.headers import *
from strategist.GOPS.baseline_models_GOPS import *
# from search_src.GOPS.value_heuristic_evaluators import GOPSValueHeuristicsSSGEvaluator
from strategist.searchlight.gameplay.simulators import GameSimulator
from strategist.GOPS.examples.abstract_list3 import abstract_list
from strategist.GOPS.examples.func_list_weiqin import gops_func_list
from strategist.Avalon.baseline_models_Avalon import *
from strategist.Avalon.examples.avalon_func_weiqin import avalon_func_list
from strategist.searchlightimprove.value_heuristic_improve import ValueHeuristicsSSGEvaluator
from strategist.utils import setup_logging_environment
from strategist.searchlightimprove.rl_evolver import RLValueHeuristicsSSGEvaluator

import logging
import os
import datetime

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import plotly.express as px

import torch as T
import torch.nn as nn
import torch.nn.functional as F


class RLValueHeuristic(ValueHeuristic2):
    # TODO: implement this class
    def __init__(self, torch_model):
        self.model = torch_model
        super().__init__()

    def _evaluate(self, state: State) -> tuple[dict[Any, float], dict]:
        values = self.model.forward(state)
        res = dict()
        for i in range(self.model.num_player):
            res[i] = float(values[i])
        notes = dict()
        # print(values)
        # print(res)
        return tuple([res, notes])


class VNetwork_GOPS(nn.Module):
    def __init__(self, input_dims=4, fc1_dims=64, fc2_dims=64, num_player=2):
        super(VNetwork_GOPS, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc1 = nn.Linear(self.input_dims * 3, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, num_player)
        self.num_player = num_player

    def forward(self, observation):
        # print(observation)
        obs_prize_cards = list(observation.prize_cards)
        while len(obs_prize_cards) < self.input_dims:
            obs_prize_cards.append(0)

        obs_player_cards = list(observation.player_cards)
        while len(obs_player_cards) < self.input_dims:
            obs_player_cards.append(0)

        obs_opponent_cards = list(observation.opponent_cards)
        while len(obs_opponent_cards) < self.input_dims:
            obs_opponent_cards.append(0)

        state = np.array(obs_prize_cards + obs_player_cards + obs_opponent_cards)
        # print(state)
        state = T.Tensor(np.array(state))
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class VNetwork_Avalon(nn.Module):
    def __init__(self, input_dims=26, fc1_dims=64, fc2_dims=64, num_player=5):
        super(VNetwork_Avalon, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, num_player)
        self.num_player = num_player

    def forward(self, observation):
        # print(observation)
        quest_leader = [observation.quest_leader]
        phase = [observation.phase]
        turn = [observation.turn]
        round = [observation.round]
        done = [observation.done]
        good_victory = [observation.good_victory]

        team_votes = list(observation.team_votes)
        # print(team_votes)
        if team_votes == []:
            team_votes = [0 for i in range(self.num_player)]

        quest_team = list(observation.quest_team)
        while len(quest_team) < 5:
            quest_team.append(-2)

        quest_votes = list(observation.quest_votes)
        while len(quest_votes) < 5:
            quest_votes.append(-2)

        quest_results = list(observation.quest_results)
        quest_results_num = []
        for quest in quest_results:
            if quest == True:
                quest_results_num.append(1)
            else:
                quest_results_num.append(0)
        while len(quest_results_num) < 5:
            quest_results_num.append(-2)

        state = np.array(quest_leader + phase + turn + round + done + good_victory + team_votes + quest_team +
                         quest_votes + quest_results_num)
        state = T.Tensor(np.array(state))
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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
    num_random_rollouts = cfg.preset_batch_evaluate.get("num_random_rollouts", 1)

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
        functions = gops_func_list
        function_categories = ['function 0', 'function 1', 'function 2', 'function 3', 'function 4',
                               'function 5', 'function 6', 'function 7', 'function 8', 'function 9']

        # TODO: add rl models
        state_dim = GOPS_num_cards
        num_players = 2
        fc1_dims = 64
        fc2_dims = 64

        rl_models = []
        rl_model_categories = []
        rl_model_names = []
        for random_seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            model = VNetwork_GOPS(input_dims=state_dim, fc1_dims=fc1_dims, fc2_dims=fc2_dims,
                                       num_player=num_players)
            state_dict = T.load('./10models_gops_5cards/rl_model_GOPS_seed{}.pt'.format(random_seed))
            model.load_state_dict(state_dict)
            rl_models.append(model)

            rl_model_categories.append('RL{}'.format(random_seed))
            rl_model_names.append('MC{}'.format(random_seed))

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
        function_categories = ['LLM'] * len(avalon_func_list)

        # TODO: add rl models
        state_dim = 21 + num_players
        fc1_dims = 128
        fc2_dims = 128

        rl_models = []
        rl_model_categories = []
        rl_model_names = []
        for random_seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            model = VNetwork_Avalon(input_dims=state_dim, fc1_dims=fc1_dims, fc2_dims=fc2_dims,
                                    num_player=num_players)
            state_dict = T.load('./10models_avalon_six_players/rl_model_Avalon_seed{}.pt'.format(random_seed))
            model.load_state_dict(state_dict)
            rl_models.append(model)

            rl_model_categories.append('RL{}'.format(random_seed))
            rl_model_names.append('MC{}'.format(random_seed))
    else:
        raise ValueError(f'Environment {env_name} not supported')

    # create evaluator for the functions
    func_evaluator = ValueHeuristicsSSGEvaluator(
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

    # create evaluator for the rl models
    rl_evaluator = RLValueHeuristicsSSGEvaluator(
        simulator=simulator,
        transitor=transitor,
        actor_enumerator=actor_enumerator,
        action_enumerator=action_enumerator,
        num_batch_runs=num_batch_runs,
        players=players,
        rng=rng,
        against_benchmark=against_benchmark,
        search_budget=num_search_budget,
        random_rollouts=num_random_rollouts
    )


    # now create both the function and rl agents
    agents = []
    agent_names = []
    agent_categories = []

    # create agents for the functions
    function_agents = func_evaluator.create_agents(functions)
    agents.extend(function_agents)
    agent_names.extend(functions)
    agent_categories.extend(function_categories)

    # create agents for the rl models
    rl_agents = rl_evaluator.create_agents(rl_models)
    agents.extend(rl_agents)
    agent_names.extend(rl_model_names)
    agent_categories.extend(rl_model_categories)

    # create agents for benchmark agents
    benchmark_names, benchmark_agents = func_evaluator.create_benchmark_agents()
    agents.extend(benchmark_agents)
    agent_names.extend(benchmark_names)
    agent_categories.extend(['Benchmark'] * len(benchmark_names))

    # evaluate the agents
    all_scores, all_notes = func_evaluator.evaluate_agents(agents=agents)

    # log time taken to evaluate functions
    endtime = datetime.datetime.now()
    logger.info(f'Time taken to evaluate functions: {endtime - starttime}')

    # store the results in a list of dictionaries
    results = []
    for name, score, category in zip(agent_names, all_scores, agent_categories):
        # append info dictionary along with the final score and function and estimated score
        to_append = {'name': name, 'score': score, 'category': category}
        results.append(to_append)

    # sort results by final score
    results = sorted(results, key=lambda x: x['score'], reverse=True)

    # log the estimated scores, final score, generation, and iteration for each function
    for item in results:
        logger.info(f'Name: {item["name"]}, Score: {item["score"]}, Category: {item["category"]}')

    # convert results to pd.DataFrame, then save to csv
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(save_dir, 'results.csv'), index=False)

    # create a plotly express box plot where x is the category and y is the score
    # add a line for each benchmark score
    # include appropriate labels and title
    fig = px.box(results_df, x='category', y='score', title='Function Scores by Category')
    # for benchmark_name, benchmark_score in benchmark_scores.items():
    #     fig.add_hline(y=benchmark_score, line_dash='dash', line_color='red', annotation_text=f'{benchmark_name} benchmark', annotation_position='bottom right')
    fig.update_layout(xaxis_title='Category', yaxis_title='Score')
    fig.write_html(os.path.join(save_dir, 'results.html'))





if __name__ == '__main__':
    setup_logging_environment()

    # Log a test message
    logging.info('This is a test log message.')

    main()