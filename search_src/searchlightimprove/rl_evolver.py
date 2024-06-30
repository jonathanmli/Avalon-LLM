from .headers import *
from searchlight.bandit import MultiarmedBanditLearner
from .llm_utils.llm_api_models import LLMModel
# from .prompts.improvement_prompts import gen_specific_improvement_prompt, gen_draw_conclusions_from_feedback_prompt, gen_implement_function_from_improvement_prompt
from .prompts.prompt_generators import PromptGenerator
from searchlight.utils import UpdatablePriorityDictionary
from searchlight.gameplay.agents import Agent
from searchlight.headers import State, ValueHeuristic2
from search_src.searchlightimprove.value_heuristic_improve import *
from search_src.GOPS.baseline_models_GOPS import *
from search_src.Avalon.baseline_models_Avalon import *
import logging
import numpy as np
import os
from typing import Optional
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import matplotlib.pyplot as plt
from search_src.searchlight.datastructures.graphs import ValueGraph2, PartialValueGraph


class RLValueHeuristicsSSGEvaluator(SimulateSearchGameEvaluator):

    def __init__(self, simulator: GameSimulator, transitor: ForwardTransitor2, actor_enumerator: ActorEnumerator, action_enumerator: ActionEnumerator, num_batch_runs: int = 10, players={0, 1, 2, 3, 4}, rng: np.random.Generator = np.random.default_rng(), against_benchmark=False, search_budget=16, random_rollouts=16, partial_information=False, stochastic_combinations=True):
        super().__init__(simulator, num_batch_runs, players, rng, stochastic_combinations=stochastic_combinations)
        self.against_benchmark = False  # against_benchmark
        self.transitor = transitor
        self.actor_enumerator = actor_enumerator
        self.action_enumerator = action_enumerator
        self.action_predictor = PolicyPredictor()
        self.search_budget = search_budget
        self.random_rollouts = random_rollouts
        self.players = players

        if not partial_information:
            self.value_graph_class = ValueGraph2
        else:
            self.value_graph_class = PartialValueGraph


    def evaluate(self, models) -> tuple[list[float], list]:
        '''
        Args:
            functions: list of functions to evaluate

        Returns:
            scores: list of scores for each function
            notes: list of notes for each function. specific notes are stored in a dictionary. 
                - if the function is not executable, the dictionary will contain an 'execution_error' key
                - otherwise the dictionary will contain the usual trajectory notes
        '''
        agents = self.create_agents(models)
        agents = self.add_filler_agents(agents)

        if not self.against_benchmark:
            # evaluate the passed functions
            passed_scores, passed_notes = super().evaluate_agents(agents)
        else:
            passed_scores = []
            passed_notes = []
            for model in models:
                # evaluate each function against the benchmark instead
                function_scores, function_notes, benchmark_scores = self.evaluate_with_benchmark([model])
                passed_scores.extend(function_scores)
                passed_notes.extend(function_notes)

        return passed_scores, passed_notes
        
    
    def create_agents(self, models) -> list[SearchAgent]:
        # create graphs
        graphs = [self.value_graph_class(adjuster=PUCTAdjuster(), utility_estimator=UtilityEstimatorLast(), players=self.players) for _ in range(len(models))]
        # create value heuristics
        value_heuristics = [RLValueHeuristic(torch_model=model) for model in models]
        # create initial inferencers
        initial_inferencers = [PackageInitialInferencer(self.transitor, self.action_enumerator, self.action_predictor, self.actor_enumerator, value_heuristic) for value_heuristic in value_heuristics]
        # create MCTS search algorithms
        search_algorithms = [SMMonteCarlo(initial_inferencer=initial_inferencer, rng=self.random_agent.rng, node_budget=self.search_budget, num_rollout=self.search_budget) for initial_inferencer in initial_inferencers]
        # create agents
        agents = [SearchAgent(search_algorithm, graph, self.rng) for graph, search_algorithm in zip(graphs, search_algorithms)]
        return agents
    
    def create_benchmark_agents(self) -> tuple[list[str], list[SearchAgent]]:
        '''
        Creates benchmark agents for evaluation
        '''
        num_agents = 1
        # create graphs
        graphs = [self.value_graph_class(adjuster=PUCTAdjuster(), utility_estimator=UtilityEstimatorLast(), players=self.players) for _ in range(num_agents)]
        # create value heuristics
        value_heuristics = [RandomRolloutValueHeuristic(self.actor_enumerator, self.action_enumerator, self.transitor, num_rollouts=self.random_rollouts, rng=self.random_agent.rng) for _ in range(num_agents)]
        # create initial inferencers
        initial_inferencers = [PackageInitialInferencer(self.transitor, self.action_enumerator, self.action_predictor, self.actor_enumerator, value_heuristic) for value_heuristic in value_heuristics]
        # create MCTS search algorithms
        search_algorithms = [SMMonteCarlo(initial_inferencer=initial_inferencer, rng=self.random_agent.rng, node_budget=self.search_budget, num_rollout=self.search_budget) for initial_inferencer in initial_inferencers]
        # create agents
        agents = [SearchAgent(search_algorithm, graph, self.rng) for graph, search_algorithm in zip(graphs, search_algorithms)]
        return ['RandomRolloutValueHeuristicMCTS'], agents

    def get_filler_agent(self) -> SearchAgent:
        '''
        Returns a filler agent that does not correspond to any function.
        '''
        # return new random agent
        graph = self.value_graph_class(players=self.players, adjuster=PUCTAdjuster(),
                                       utility_estimator=UtilityEstimatorLast())
        value_heuristic = RandomRolloutValueHeuristic(actor_enumerator=self.actor_enumerator,
                                                      action_enumerator=self.action_enumerator,
                                                      forward_transitor=self.transitor,
                                                      num_rollouts=self.random_rollouts, rng=self.random_agent.rng,
                                                      players=self.players)
        initial_inferencer = PackageInitialInferencer(self.transitor, self.action_enumerator, self.action_predictor,
                                                      self.actor_enumerator, value_heuristic)
        search_algorithm = SMMonteCarlo(initial_inferencer=initial_inferencer, rng=self.random_agent.rng,
                                        node_budget=self.search_budget, num_rollout=self.search_budget)
        return SearchAgent(search_algorithm, graph, self.rng)

    def add_filler_agents(self, agents: list[SearchAgent]) -> list[SearchAgent]:
        '''
        Adds filler agents to the list of agents so that we have enough agents to play the game. Filler agents are agents that do not correspond to any function.
        '''
        # num_to_add should be number of [players - number of agents]^+
        num_to_add = max(0, len(self.players) - len(agents))

        filler_agents = [self.get_filler_agent() for _ in range(num_to_add)]
        return agents + filler_agents


    def evaluate_with_benchmark(self, models) -> tuple[list[float], list, dict[str, float]]:
        '''
        Evaluates the functions with additional benchmark agents. The benchmark agents are:
        - RandomRolloutValueHeuristic under MCTS search

        Also possibly increases search budget for the agents. 

        Functions are assumed to be executable.

        Args:
            functions: list of functions to evaluate

        Returns:
            function_scores: list of scores for each function
            notes: list of notes for each function. specific notes are stored in a dictionary. 
                - if the function is not executable, the dictionary will contain an 'execution_error' key
                - otherwise the dictionary will contain the usual trajectory notes
            benchmark_scores: dictionary of benchmark scores for each benchmark agent
        '''
        # first create the function agents
        model_agents = self.create_agents(models)

        # create benchmark agents
        benchmark_names, benchmark_agents = self.create_benchmark_agents()

        all_agents = model_agents + benchmark_agents
        # add filler agents if necessary
        all_agents = self.add_filler_agents(all_agents)

        # print(f'evaluating {len(all_agents)} agents {all_agents}')
        # run the evaluation
        scores, notes = self.evaluate_agents(all_agents)

        # print('done evaluating agents')

        # separate the scores
        function_scores = scores[:len(model_agents)]
        benchmark_scores = scores[len(model_agents):]
        function_notes = notes[:len(model_agents)]

        # assign the benchmark scores to the benchmark names
        benchmark_scores = {name: score for name, score in zip(benchmark_names, benchmark_scores)}

        return function_scores, function_notes, benchmark_scores


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
    def __init__(self, input_dims=5, fc1_dims=64, fc2_dims=64, num_player=2):
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



class RLEvolver(Evolver):
    '''
    Abstract class for evolving functions
    '''

    # functions_dict: UpdatablePriorityDictionary # where values are a dictionary

    def __init__(self, config, evaluator: Evaluator, batch_size: int = 10, rd_seed=0,):
        '''
        Args:
            evaluator: evaluator for functions
            batch_size: number of functions to propose at a time
            seed_functions: list of seed functions to start with, (function, abstract)
            check_function: function to check if a function is valid
            parse_function: function to parse a function
            model: LLM model to use for generating functions
            num_fittest_functions: number of fittest functions to consider each iteration
        '''
        super().__init__()
        self.config = config
        self.evaluator = evaluator
        self.batch_size = batch_size
        self.num_evolutions = 0  # number of evolutions conducted
        self.data = []  # data for training the model
        self.rd_seed = rd_seed

        self.env_name = self.config.env_preset.env_name
        if self.env_name == 'GOPS':
            self.state_dim = self.config.env_preset.num_cards
            self.num_players = 2
            self.fc1_dims = 64
            self.fc2_dims = 64
            self.lr_V = 8e-4

            self.model = VNetwork_GOPS(input_dims=self.state_dim, fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims,
                                       num_player=self.num_players)
            self.V_optimizer = optim.SGD(self.model.parameters(), lr=self.lr_V)
        elif self.env_name == 'Avalon':
            self.num_players = self.config.env_preset.num_players
            self.state_dim = 21 + self.num_players
            self.fc1_dims = 128
            self.fc2_dims = 128
            self.lr_V = 5e-4

            self.model = VNetwork_Avalon(input_dims=self.state_dim, fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims,
                                       num_player=self.num_players)
            self.V_optimizer = optim.SGD(self.model.parameters(), lr=self.lr_V)

        self.performance_history = []
        # do initial evaluation of your model
        #  notes[0]['trajectory_data'] is the list containing num_batch_runs dictionaries
        scores, notes = self.evaluate([self.model])
        # print(notes[0]['trajectory_data'])
        self.data.append(notes[0]['trajectory_data'])
        # self.performance_history.append(scores[0])
        # self.num_evolutions += 1

        # create logger
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def evaluate(self, models) -> tuple[list[float], list[dict]]:
        return self.evaluator.evaluate(models)
    
    def evolve_once(self):
        '''
        Conducts one cycle of evolution
        '''

        training_data = self.data[-1]
        # print('len_last_data', len(training_data))  # len = num_batch_runs

        for tra in range(len(training_data)):  # len = num_batch_runs
            data_trajectory = training_data[tra]['trajectory']  # a list of num_transitions
            score_trajectory = training_data[tra]['score_trajectory']  # a list of num_transitions
            # score_trajectory = training_data[tra]['search_trajectory']  # a list of num_transitions

            # train the model
            for i in range(len(data_trajectory)):  # num_transitions in one episode
                state = list(data_trajectory[i])[-1]

                if self.env_name == 'GOPS':
                    score_lst = [score_trajectory[i][0], score_trajectory[i][1]]
                elif self.env_name == 'Avalon':
                    score_lst = []
                    for num_play in range(self.num_players):
                        score_lst.append(score_trajectory[i][num_play])

                # learn VNetwork
                self.V_optimizer.zero_grad()
                V_pred = self.model.forward(state)
                V_true = T.FloatTensor(score_lst)
                loss_fn = nn.MSELoss()
                V_loss = loss_fn(V_pred, V_true)
                V_loss.backward()
                self.V_optimizer.step()

        # self.logger.info('training data: {training_data}')
        # evaluate the model
        #  notes[0]['trajectory_data'] is the list containing num_batch_runs dictionaries
        scores, notes = self.evaluate([self.model])
        # print('notes_trajectory_data111 =', len(notes[0]['trajectory_data']))  # num_batch_runs dictionaries
        self.performance_history.append(scores[0])

        self.num_evolutions += 1
        self.data.append(notes[0]['trajectory_data'])


    def evolve(self, num_cycles: int):
        '''
        Evolves the functions for a certain number of cycles
        '''
        for _ in range(num_cycles):
            self.evolve_once()
        print('Task: {}, \t Seed: {}, \t Performance History = {}'.format(self.env_name, self.rd_seed, self.performance_history))
        np.save('performance_history_{}_seed{}.npy'.format(self.env_name, self.rd_seed), self.performance_history)
        T.save(self.model.state_dict(), 'rl_model_{}_seed{}.pt'.format(self.env_name, self.rd_seed))

    def produce_analysis(self):
        '''
        Produces an analysis of the evolution
        '''
        scores, notes = self.evaluate([self.model])

        self.logger.info(f'Final score: {scores[0]}')