from .headers import Evaluator
from strategist.searchlight.gameplay.simulators import GameSimulator
from strategist.searchlight.headers import *
from strategist.searchlight.gameplay.agents import SearchAgent
from strategist.searchlight.algorithms.mcts_search import SMMonteCarlo
from strategist.searchlight.datastructures.graphs import ValueGraph2, PartialValueGraph
from strategist.searchlight.datastructures.adjusters import PUCTAdjuster
from strategist.searchlight.datastructures.estimators import UtilityEstimatorMean
from strategist.searchlight.classic_models import RandomRolloutValueHeuristic
from .llm_utils.llm_api_models import LLMModel
from .evaluators import *
from typing import Type
from strategist.searchlightimprove.prompts.prompt_generators import PromptGenerator

class LLMFuncValueHeuristic(ValueHeuristic2):

    def __init__(self, func: str):
        pass

class ValueHeuristicsSSGEvaluator(SimulateSearchGameEvaluator):

    def __init__(self, simulator: GameSimulator, transitor: ForwardTransitor2, actor_enumerator: ActorEnumerator, action_enumerator: ActionEnumerator, check_function, llm_func_value_heuristic_class: Type[LLMFuncValueHeuristic], players, num_batch_runs: int = 10,  rng: np.random.Generator = np.random.default_rng(), against_benchmark=False, search_budget=16, random_rollouts=16, partial_information=False, stochastic_combinations = True, filler_heuristic: Optional[ValueHeuristic2] = None, use_filler_as_benchmark: bool =True, set_filler_to_best_func: bool = True):
        super().__init__(simulator, num_batch_runs, players=players, rng=rng, stochastic_combinations=stochastic_combinations)
        self.against_benchmark = against_benchmark
        self.transitor = transitor
        self.actor_enumerator = actor_enumerator
        self.action_enumerator = action_enumerator
        self.action_predictor = PolicyPredictor()
        self.check_function = check_function
        self.llm_func_value_heuristic_class = llm_func_value_heuristic_class
        self.search_budget = search_budget
        self.random_rollouts = random_rollouts
        self.filler_heuristic = filler_heuristic
        self.use_filler_as_benchmark = use_filler_as_benchmark
        self.set_filler_to_best_func = set_filler_to_best_func

        if not partial_information:
            self.value_graph_class = ValueGraph2
        else:
            self.value_graph_class = PartialValueGraph

    def set_filler_heuristic(self, filler_heuristic: Optional[ValueHeuristic2]):
        self.filler_heuristic = filler_heuristic

    def evaluate(self, functions: list[str]) -> tuple[list[float], list]:
        '''
        Args:
            functions: list of functions to evaluate

        Returns:
            scores: list of scores for each function
            notes: list of notes for each function. specific notes are stored in a dictionary. 
                - if the function is not executable, the dictionary will contain an 'execution_error' key
                - otherwise the dictionary will contain the usual trajectory notes
        '''
        # check that all the functions are executable before passing to super().evaluate
        passed_functions = []
        unpassed_notes = []
        for i, func in enumerate(functions):
            try:
                self.logger.debug(f'Checking function {self.check_function}')
                self.check_function(func, False)
                passed_functions.append(func)
                # TODO: maybe don't need this
            except Exception as e:
                unpassed_notes.append({'execution_error': e})

        if not self.against_benchmark:
            agents = self.create_agents(passed_functions)
            # add filler agents if necessary
            agents = self.add_filler_agents(agents)
            # evaluate the passed functions
            passed_scores, passed_notes = super().evaluate_agents(agents)
            passed_scores = passed_scores[:len(passed_functions)]
            pass_notes = passed_notes[:len(passed_functions)]
        else:
            passed_scores = []
            passed_notes = []
            # evaluate each function against the benchmark instead
            for func in passed_functions:
                function_scores, function_notes, benchmark_scores = self.evaluate_with_benchmark([func])
                passed_scores.append(function_scores[0])
                passed_notes.append(function_notes[0])

        # if set_filler_to_best_func is True, set the filler heuristic to the best passed function
        if self.set_filler_to_best_func:
            best_index = np.argmax(passed_scores)
            best_func = passed_functions[best_index]
            best_value_heuristic = self.llm_func_value_heuristic_class(func=best_func)
            self.set_filler_heuristic(best_value_heuristic)

        # print('passed_scores', passed_scores)
        # print('passed_notes', passed_notes)

        # combine passed and unpassed notes such that the indices match the functions
        notes = []  
        passed_index = 0
        unpassed_index = 0
        scores = [float('-inf')] * len(functions)
        for i, func in enumerate(functions):
            if func in passed_functions:
                notes.append(passed_notes[passed_index])
                scores[i] = passed_scores[passed_index]
                passed_index += 1
            else:
                notes.append(unpassed_notes[unpassed_index])
                unpassed_index += 1

        return scores, notes
        
    
    def create_agents(self, functions: list[str]) -> list[SearchAgent]:
        # create graphs
        graphs = [self.value_graph_class(players=self.players, adjuster=PUCTAdjuster(), utility_estimator=UtilityEstimatorMean()) for _ in range(len(functions))]
        # create value heuristics
        value_heuristics = [self.llm_func_value_heuristic_class(func=func) for func in functions]
        # create initial inferencers
        initial_inferencers = [PackageInitialInferencer(self.transitor, self.action_enumerator, self.action_predictor, self.actor_enumerator, value_heuristic) for value_heuristic in value_heuristics]
        # create MCTS search algorithms
        search_algorithms = [SMMonteCarlo(initial_inferencer=initial_inferencer, rng=self.random_agent.rng, node_budget=self.search_budget, num_rollout=self.search_budget) for initial_inferencer in initial_inferencers]
        # create agents
        agents = [SearchAgent(search_algorithm, graph, self.rng) for graph, search_algorithm in zip(graphs, search_algorithms)]
        # return []
        return agents
    
    def create_benchmark_agents(self) -> tuple[list[str], list[SearchAgent]]:
        '''
        Creates benchmark agents for evaluation
        '''
        num_agents = 1
        # create graphs
        graphs = [self.value_graph_class(players=self.players, adjuster=PUCTAdjuster(), utility_estimator=UtilityEstimatorMean()) for _ in range(num_agents)]

        # create value heuristics
        if self.use_filler_as_benchmark and self.filler_heuristic is not None:
            value_heuristic = self.get_filler_heuristic()
            names = ['FillerHeuristicMCTS' for _ in range(num_agents)]
            value_heuristics = [value_heuristic for _ in range(num_agents)]
        else:
            value_heuristics = [RandomRolloutValueHeuristic(actor_enumerator=self.actor_enumerator, action_enumerator=self.action_enumerator, forward_transitor=self.transitor, num_rollouts=self.random_rollouts, rng=self.random_agent.rng,  players= self.players) for _ in range(num_agents)]
            names = ['RandomRolloutValueHeuristicMCTS' for _ in range(num_agents)]
        # create initial inferencers
        initial_inferencers = [PackageInitialInferencer(self.transitor, self.action_enumerator, self.action_predictor, self.actor_enumerator, value_heuristic) for value_heuristic in value_heuristics]
        # create MCTS search algorithms
        search_algorithms = [SMMonteCarlo(initial_inferencer=initial_inferencer, rng=self.random_agent.rng, node_budget=self.search_budget, num_rollout=self.search_budget) for initial_inferencer in initial_inferencers]
        # create agents
        agents = [SearchAgent(search_algorithm, graph, self.rng) for graph, search_algorithm in zip(graphs, search_algorithms)]
        return names, agents
    
    def get_filler_heuristic(self) -> ValueHeuristic2:
        '''
        Returns a filler heuristic that does not correspond to any function. 
        '''
        if self.filler_heuristic is None:
            return RandomRolloutValueHeuristic(actor_enumerator=self.actor_enumerator, action_enumerator=self.action_enumerator, forward_transitor=self.transitor, num_rollouts=self.random_rollouts, rng=self.random_agent.rng,  players= self.players)
        else:
            # log the filler heuristic for debugging
            self.logger.info(f'Using filler heuristic {self.filler_heuristic}')
            return self.filler_heuristic
    
    def get_filler_agent(self) -> SearchAgent:
        '''
        Returns a filler agent that does not correspond to any function. 
        '''
        # return new random agent 
        graph = self.value_graph_class(players=self.players, adjuster=PUCTAdjuster(), utility_estimator=UtilityEstimatorMean())
        value_heuristic = self.get_filler_heuristic()
        initial_inferencer = PackageInitialInferencer(self.transitor, self.action_enumerator, self.action_predictor, self.actor_enumerator, value_heuristic)
        search_algorithm = SMMonteCarlo(initial_inferencer=initial_inferencer, rng=self.random_agent.rng, node_budget=self.search_budget, num_rollout=self.search_budget)
        return SearchAgent(search_algorithm, graph, self.rng)
        

    def add_filler_agents(self, agents: list[SearchAgent]) -> list[SearchAgent]:
        '''
        Adds filler agents to the list of agents so that we have enough agents to play the game. Filler agents are agents that do not correspond to any function. 
        '''
        # num_to_add should be number of [players - number of agents]^+
        num_to_add = max(0, len(self.players) - len(agents))

        filler_agents = [self.get_filler_agent() for _ in range(num_to_add)]
        return agents + filler_agents 
    
    def evaluate_with_benchmark(self, functions: list[str]) -> tuple[list[float], list, dict[str, float]]:
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
        function_agents = self.create_agents(functions)

        # create benchmark agents
        benchmark_names, benchmark_agents = self.create_benchmark_agents()

        all_agents = function_agents + benchmark_agents

        # add filler agents if necessary
        all_agents = self.add_filler_agents(all_agents)

        # print(f'evaluating {len(all_agents)} agents {all_agents}')
        # run the evaluation
        scores, notes = self.evaluate_agents(all_agents)

        # print('done evaluating agents')

        # separate the scores
        function_scores = scores[:len(functions)]
        benchmark_scores = scores[len(functions):len(functions)+len(benchmark_agents)]
        function_notes = notes[:len(functions)]

        # assign the benchmark scores to the benchmark names
        benchmark_scores = {name: score for name, score in zip(benchmark_names, benchmark_scores)}

        return function_scores, function_notes, benchmark_scores
    

class LLMCriticValueHeuristicEvaluator(Evaluator):

    def __init__(self, llm_model: LLMModel, prompt_generator: PromptGenerator):
        super().__init__()
        self.llm_model = llm_model
        self.prompt_generator = prompt_generator

    def evaluate(self, functions: list[str]) -> tuple[list[float], list]:
        notes = []
        scores = []
        for func in functions:
            score, note = self.evaluate_single_function(func)
            notes.append(note)
            scores.append(score)
        return scores, notes

    def evaluate_single_function(self, func: str) -> tuple[float, dict]:
        prompt = self.prompt_generator.gen_critic_score_prompt(func)
        response = self.llm_model.generate(prompt)[0]
        explanation, score = self.prompt_generator.parse_critic_score(response)
        note = {'explanation': explanation, "score":score}
        return score, note