from typing import Type
from matplotlib.pylab import Generator
from numpy.random._generator import default_rng as default_rng
from search_src.searchlight.gameplay.simulators import GameSimulator
from search_src.searchlight.headers import ActionEnumerator, ActorEnumerator, ForwardTransitor2
from search_src.searchlightimprove.evaluators import ActionEnumerator, ActorEnumerator, ForwardTransitor2
from search_src.searchlightimprove.headers import Evaluator
from search_src.GOPS.baseline_models_GOPS import *
from search_src.searchlight.classic_models import *
# from search_src.searchlightimprove.llm_utils.llm_api_models import GPT35
from search_src.searchlight.headers import *
from search_src.searchlight.datastructures.graphs import *
from search_src.searchlight.datastructures.adjusters import *
from search_src.searchlight.datastructures.beliefs import *
from search_src.searchlight.datastructures.estimators import *
from search_src.searchlight.algorithms.mcts_search import *
from search_src.searchlight.algorithms.full_search import SMMinimax
from search_src.searchlight.gameplay.agents import SearchAgent
from search_src.searchlight.gameplay.simulators import *
from search_src.searchlightimprove.evaluators import *
from search_src.searchlightimprove.value_heuristic_improve import LLMFuncValueHeuristic, ValueHeuristicsSSGEvaluator

import numpy as np
from collections import defaultdict
import itertools

class RandomRolloutEvaluator(Evaluator):
    '''
    Compares the out of the generated function to that of random rollout heuristic on randomly sampled states
    '''

    # TODO: this needs some unit tests

    def __init__(self, sample_size: int = 10):
        super().__init__()
        # create config
        random_state = np.random.RandomState(12)
        action_enumerator = GOPSActionEnumerator()
        forward_transitor = GOPSForwardTransitor2()
        actor_enumerator = GOPSActorEnumerator()
        self.value_heuristic = RandomRolloutValueHeuristic(actor_enumerator, action_enumerator,
                                                        forward_transitor, num_rollouts=100, 
                                                        random_state=random_state)
        GOPS_initial_state = GOPSState2(
            {-1},
            prize_cards=tuple(),
            player_cards=tuple(),
            opponent_cards=tuple(),
            num_cards=6
        )
        self.samples_states = self.value_heuristic.sample_states(GOPS_initial_state)
        self.sample_size = sample_size
        self.random_state = random_state
        
    def evaluate(self, functions: list[str]) -> tuple[list[float], list]:
        scores = []

        # randomly select sample_size number of states from self.samples_states
        sample_indices = self.random_state.choice(len(self.samples_states), self.sample_size, replace=True)
        sample_states = [self.samples_states[i] for i in sample_indices]

        for func in functions:

            # print('func', func)
            # create LLM value heuristic
            llm_value_heuristic = LLMFunctionalValueHeuristic(func = func)

            # if not llm_value_heuristic.passed, then add 0 to scores
            if not llm_value_heuristic.passed:
                scores.append(0)
            else:
                # compare the two value heuristics across sample
                score = 0
                for state in sample_states:
                    score += abs(self.value_heuristic.evaluate(state) - llm_value_heuristic.evaluate(state))
                score = score / self.sample_size
                scores.append(-score)
                # print('score', score)
        # print('scores', scores)
        return scores, scores

# NOTE: deprecated
# class GOPSValueHeuristicsSSGEvaluator(SimulateSearchGameEvaluator):

#     def __init__(self, simulator: GameSimulator, num_batch_runs: int = 10, players = {0,1}, rng: np.random.Generator = np.random.default_rng(), against_benchmark=False, search_budget=16, random_rollout_num_rollouts=32,):
#         super().__init__(simulator, num_batch_runs, players, rng)
#         self.against_benchmark = against_benchmark
#         self.transitor = GOPSForwardTransitor2()
#         self.actor_enumerator = GOPSActorEnumerator()
#         self.action_enumerator = GOPSActionEnumerator()
#         self.action_predictor = PolicyPredictor()
#         self.check_function = LLMFunctionalValueHeuristic.test_evaluate_static
#         self.intermediate_search_budget = search_budget
#         self.random_rollout_num_rollouts = random_rollout_num_rollouts

#     def set_seach_budget(self, search_budget: int):
#         self.intermediate_search_budget = search_budget

#     def evaluate(self, functions: list[str]) -> tuple[list[float], list]:
#         '''
#         Args:
#             functions: list of functions to evaluate

#         Returns:
#             scores: list of scores for each function
#             notes: list of notes for each function. specific notes are stored in a dictionary. 
#                 - if the function is not executable, the dictionary will contain an 'execution_error' key
#                 - otherwise the dictionary will contain the usual trajectory notes
#         '''
#         # check that all the functions are executable before passing to super().evaluate
#         passed_functions = []
#         unpassed_notes = []
#         for i, func in enumerate(functions):
#             try:
#                 self.check_function(func, False)
#                 passed_functions.append(func)
#             except Exception as e:
#                 unpassed_notes.append({'execution_error': e})

#         if not self.against_benchmark:
#             # evaluate the passed functions
#             passed_scores, passed_notes = super().evaluate(passed_functions)
#         else:
#             passed_scores = []
#             passed_notes = []
#             # evaluate each function against the benchmark instead
#             for func in passed_functions:
#                 function_scores, function_notes, benchmark_scores = self.evaluate_with_benchmark([func])
#                 passed_scores.extend(function_scores)
#                 passed_notes.extend(function_notes)

#         # print('passed_scores', passed_scores)
#         # print('passed_notes', passed_notes)

#         # combine passed and unpassed notes such that the indices match the functions
#         notes = []  
#         passed_index = 0
#         unpassed_index = 0
#         scores = [float('-inf')] * len(functions)
#         for i, func in enumerate(functions):
#             if func in passed_functions:
#                 notes.append(passed_notes[passed_index])
#                 scores[i] = passed_scores[passed_index]
#                 passed_index += 1
#             else:
#                 notes.append(unpassed_notes[unpassed_index])
#                 unpassed_index += 1

#         return scores, notes
        
    
#     def create_agents(self, functions: list[str]) -> list[SearchAgent]:
#         # create graphs
#         graphs = [ValueGraph2(adjuster=PUCTAdjuster(), utility_estimator=UtilityEstimatorLast()) for _ in range(len(functions))]
#         # create value heuristics
#         value_heuristics = [LLMFunctionalValueHeuristic(None, func) for func in functions]
#         # create initial inferencers
#         initial_inferencers = [GOPSInitialInferencer2(self.transitor, self.action_enumerator, self.action_predictor, self.actor_enumerator, value_heuristic) for value_heuristic in value_heuristics]
#         # create MCTS search algorithms
#         search_algorithms = [SMMonteCarlo(initial_inferencer=initial_inferencer, rng=self.random_agent.rng, node_budget=self.intermediate_search_budget, num_rollout=self.intermediate_search_budget) for initial_inferencer in initial_inferencers]
#         # create agents
#         agents = [SearchAgent(search_algorithm, graph, self.rng) for graph, search_algorithm in zip(graphs, search_algorithms)]
#         return agents
    
#     def create_benchmark_agents(self) -> tuple[list[str], list[SearchAgent]]:
#         '''
#         Creates benchmark agents for evaluation
#         '''
#         num_agents = 1
#         # create graphs
#         graphs = [ValueGraph2(adjuster=PUCTAdjuster(), utility_estimator=UtilityEstimatorLast()) for _ in range(num_agents)]
#         # create value heuristics
#         value_heuristics = [RandomRolloutValueHeuristic(self.actor_enumerator, self.action_enumerator, self.transitor, num_rollouts=self.random_rollout_num_rollouts, rng=self.random_agent.rng) for _ in range(num_agents)]
#         # create initial inferencers
#         initial_inferencers = [GOPSInitialInferencer2(self.transitor, self.action_enumerator, self.action_predictor, self.actor_enumerator, value_heuristic) for value_heuristic in value_heuristics]
#         # create MCTS search algorithms
#         search_algorithms = [SMMonteCarlo(initial_inferencer=initial_inferencer, rng=self.random_agent.rng, node_budget=self.intermediate_search_budget, num_rollout=self.intermediate_search_budget) for initial_inferencer in initial_inferencers]
#         # create agents
#         agents = [SearchAgent(search_algorithm, graph, self.rng) for graph, search_algorithm in zip(graphs, search_algorithms)]
#         return ['RandomRolloutValueHeuristicMCTS'], agents
    
#     def evaluate_with_benchmark(self, functions: list[str]) -> tuple[list[float], list, dict[str, float]]:
#         '''
#         Evaluates the functions with additional benchmark agents. The benchmark agents are:
#         - RandomRolloutValueHeuristic under MCTS search

#         Also possibly increases search budget for the agents. 

#         Functions are assumed to be executable.

#         Args:
#             functions: list of functions to evaluate

#         Returns:
#             function_scores: list of scores for each function
#             notes: list of notes for each function. specific notes are stored in a dictionary. 
#                 - if the function is not executable, the dictionary will contain an 'execution_error' key
#                 - otherwise the dictionary will contain the usual trajectory notes
#             benchmark_scores: dictionary of benchmark scores for each benchmark agent
#         '''
#         # first create the function agents
#         function_agents = self.create_agents(functions)

#         # create benchmark agents
#         benchmark_names, benchmark_agents = self.create_benchmark_agents()

#         all_agents = function_agents + benchmark_agents

#         # print(f'evaluating {len(all_agents)} agents {all_agents}')
#         # run the evaluation
#         scores, notes = self.evaluate_agents(all_agents)

#         # print('done evaluating agents')

#         # separate the scores
#         function_scores = scores[:len(function_agents)]
#         benchmark_scores = scores[len(function_agents):]
#         function_notes = notes[:len(function_agents)]

#         # assign the benchmark scores to the benchmark names
#         benchmark_scores = {name: score for name, score in zip(benchmark_names, benchmark_scores)}

#         return function_scores, function_notes, benchmark_scores

