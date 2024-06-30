from .headers import *
from searchlight.bandit import MultiarmedBanditLearner
from searchlight.utils import UpdatablePriorityDictionary
from .evolver_dependencies import *
from .strategy_libraries import *

import numpy as np
import pandas as pd
import os

from typing import Optional


class BasicEvolver(Evolver2):
    '''
    Abstract class for evolving strategies
    '''

    strategy_library: StrategyLibrary

    def __init__(self, evolver_dependency: EvolverDependency,  seed_strategies: list[tuple[Any, dict]], num_fittest_strategies: int = 1, stopping_threshold: Optional[float] = None, strategy_library_name: str = "bfs"):
        '''
        Args:
            evolver_dependency: dependency for the evolver
            seed_strategies: list of seed strategies to start with, (strategy, abstract)
            num_fittest_strategies: number of fittest strategies to consider each iteration
        '''
        super().__init__(evolver_dependency=evolver_dependency)
        self.num_evolutions = 0 # number of evolutions conducted
        self.num_fittest_strategies = num_fittest_strategies
        self.stopping_threshold = stopping_threshold

        if strategy_library_name == "bfs":
            self.strategy_library = BFSStrategyLibrary()
        elif strategy_library_name == "mcts":
            self.strategy_library = MCTSStrategyLibrary()
        else:
            raise ValueError('Invalid strategy library name')

        # add seed strategies
        self.add_seed_strategies(seed_strategies)
            
    def add_seed_strategies(self, seed_strategies: list[tuple[Any, dict]]):
        '''
        Add seed strategies to the strategies dictionary.

        Args:
            seed_strategies: list of seed strategies to add to the strategies dictionary, of the form (strategy, notes)
        '''

        for i, (strategy, strategy_info) in enumerate(seed_strategies):
            # evaluate
            score, is_successful = self.evaluate(strategy, strategy_info)
            if is_successful:
                info = {'iteration': 0, 'generation': 0, 'predecessor_strategy': None, 'idea_trace': tuple(), 'last_trajectory': tuple(), 'last_idea': None} | strategy_info
                self.add_or_update_strategy(strategy, info, score)
    
    def add_or_update_strategy(self, strategy: Any, notes: dict, score: float):
        '''
        Adds or updates a strategy in the strategies dictionary

        Args:
            strategy: strategy to add or update
            notes: notes for the strategy
            score: score of the strategy
        '''
        self.strategy_library.add_or_update_strategy(strategy, notes, score)

    def get_fittest(self, k: int = 1) -> list[tuple[Any, dict, float]]:
        '''
        Returns the k fittest items (highest to lowest). If there are less than k strategies, return all strategies

        Items of the form (strategy, dict(abstract, feedback, iteration), priority)
        '''

        # get the fittest strategies equal to the batch size
        fittest_items = self.strategy_library.select_strategies(k)
        return fittest_items
    
    def evaluate(self, strategy: Any, strategy_info: dict) -> tuple[float, bool]:
        return self.evolver_dependency.evaluate(strategy=strategy, strategy_info=strategy_info)
    
    @abstractmethod
    def evolve_once(self):
        '''
        Conducts one cycle of evolution
        '''
        pass

    def _evolve(self, num_cycles: int):
        '''
        Evolves the strategies for a certain number of cycles
        '''
        for _ in range(num_cycles):
            # if threshold is not none, check if best strategy is above threshold
            if self.stopping_threshold is not None:
                best_strategy, best_score, best_info = self.get_best_strategy()
                if best_score >= self.stopping_threshold:
                    self.logger.info(f'Stopping evolution because best strategy is above threshold: {best_score}')
                    break
            self.num_evolutions += 1
            self.evolve_once()

    def get_strategies_df(self, k: int = -1) -> pd.DataFrame:
        '''
        Returns a dataframe of the top k strategies in the strategies dictionary
        '''
        # store results as a list of dictionaries
        results = []
        top_items = self.strategy_library.select_strategies(k)
        for strategy, info, score in top_items:
            to_append = info | {'strategy': strategy, 'score': score}
            results.append(to_append)

        # convert results to a dataframe
        results_df = pd.DataFrame(results)
        return results_df

    def get_best_strategy(self):
        best_strategy, best_info, best_score = self.strategy_library.get_best_strategy()
        return best_strategy, best_score, best_info
    
class TreeEvolver(BasicEvolver):
    '''
    Class for evolving strategies using BFS
    '''

    evolver_dependency: TreeEvolverDependency

    def __init__(self, evolver_dependency: TreeEvolverDependency, seed_strategies: list[tuple[Any, dict]], batch_size: int = -1, num_fittest_strategies: int = 1, stopping_threshold: Optional[float] = None, strategy_library_name: str = "bfs"):
        super().__init__(evolver_dependency=evolver_dependency, seed_strategies=seed_strategies, num_fittest_strategies=num_fittest_strategies, stopping_threshold=stopping_threshold, strategy_library_name=strategy_library_name)
        if batch_size == -1:
            batch_size = num_fittest_strategies
        self.batch_size = batch_size

    def evolve_once(self):
        '''
        Conducts one cycle of evolution
        '''
        # get the fittest strategies 
        fittest_items = self.get_fittest(self.num_fittest_strategies)

        # propose improvements for the fittest strategies
        counter = 0
        if len(fittest_items) == 0:
            self.logger.info('No strategies in the library')
            return

        while counter < self.batch_size:
            for func, info, priority in fittest_items:
                new_func, new_note, is_generated = self.generate_strategy(func, info)
                if is_generated:
                    new_info = info | {'iteration': self.num_evolutions, 'generation': info['generation'] + 1, 'predecessor_strategy': func} | new_note 
                    # evaluate the proposed strategy
                    score, is_successful = self.evaluate(new_func, info)

                    if is_successful:
                        # store the improved strategy
                        self.add_or_update_strategy(new_func, new_info, score)
                counter += 1

    def generate_strategy(self, old_strategy: Any, old_info: dict) -> tuple[Any, dict, bool]:
        '''
        Generates a strategy given a prompt
        '''
        return self.evolver_dependency.generate_new_strategy(old_strategy=old_strategy, old_info=old_info)

class StrategistEvolver(BasicEvolver):
    '''
    Conducts evolution with a scored library (bandit learner) of improvement ideas.

    This will build upon the base Evolver, but with an additional library of improvement ideas with scores that will be used to guide the evolution.
    '''
    mbleaner: MultiarmedBanditLearner
    evolver_dependency: StrategistEvolverDependency

    def __init__(self,  evolver_dependency: EvolverDependency,  seed_strategies: list[tuple[str, dict]], batch_size: int = -1, num_fittest_strategies: int = 1, num_ideas_per_iteration: int = 2, num_ideas_per_strategy: int = 1, stopping_threshold: Optional[float] = None, strategy_library_name: str = "bfs"):
        '''
        Args:
            evolver_dependency: dependency for the evolver
            batch_size: number of strategies generate each iteration
            seed_strategies: seed strategies to start the evolution
            num_fittest_strategies: number of fittest strategies to consider each iteration
            num_ideas_per_iteration: number of ideas to generate each iteration
            num_ideas_per_strategy: number of ideas to generate per strategy
            stopping_threshold: threshold to stop evolution
        '''
        super().__init__(evolver_dependency=evolver_dependency, seed_strategies=seed_strategies, num_fittest_strategies=num_fittest_strategies, stopping_threshold=stopping_threshold, strategy_library_name=strategy_library_name)
        self.mbleaner = MultiarmedBanditLearner()
        self.num_ideas_per_iteration = num_ideas_per_iteration
        self.num_ideas_per_strategy = num_ideas_per_strategy
        if batch_size == -1:
            batch_size = num_fittest_strategies
        self.batch_size = batch_size

        # num implements per iteration should be batch_size integer divided by num_fittest_strategies
        num_implements_per_iteration = self.batch_size // self.num_fittest_strategies
        num_idea_loops = self.num_ideas_per_iteration // (self.num_fittest_strategies * self.num_ideas_per_strategy)

        # if self.batch_size % self.num_fittest_strategies is not 0, log a warning
        self.logger.info(f'Batch size: {self.batch_size}, Num fittest strategies: {self.num_fittest_strategies}, Num ideas per iteration: {self.num_ideas_per_iteration}, Num idea loops: {num_idea_loops}, Num implements per iteration: {num_implements_per_iteration}')
        if self.batch_size % self.num_fittest_strategies != 0:
            self.logger.warning(f'Batch size {self.batch_size} is not divisible by num fittest strategies {self.num_fittest_strategies}')
        # if self.num_ideas_per_iteration % self.num_fittest_strategies is not 0, log a warning
        if self.num_ideas_per_iteration %  (self.num_fittest_strategies * self.num_ideas_per_strategy) != 0:
            self.logger.warning(f'Num ideas per iteration {self.num_ideas_per_iteration} is not divisible by num fittest strategies {self.num_fittest_strategies} times num ideas per strategy {self.num_ideas_per_strategy}')
        

    def generate_improvement_ideas(self, num_fittest_strategies: int, num_total_ideas: int, num_ideas_per_strategy:int=1, improvement_prior=0.0) -> None:
        '''
        Generates improvement ideas and adds them to the bandit learner

        This is basically a reflection step where the agent reflects on the feedback and generates improvement ideas.
        
        We get the top batch_size strategies from our strategy library along with their numerical feedback. We then pass the feedback to the feedback analyzer to sample and translate the feedback to numerical form. We then ask the LLM to reflect on the feedback and generate num_ideas improvement ideas. We then add the improvement ideas to the bandit learner.

        Args:
            batch_size: number of strategies to sample from the strategy library
            num_loops: number of times to repeat the process
            num_ideas: number of improvement ideas to generate per prompt
        '''
        # get the top batch_size strategies from the strategy library
        top_items = self.get_fittest(num_fittest_strategies)

        # assert that at least 1 strategy is in the library
        if len(top_items) < 1:
            raise ValueError('No strategies in the library')

        improvement_ideas = []
        idea_notes = []
        counter = 0

        while counter < num_total_ideas:
            for strategy, strategy_info, strategy_score in top_items:
                new_ideas, new_notes = self.generate_improvement_ideas_from_strategy(strategy, strategy_info, num_ideas_per_strategy)
                # append the new ideas and notes until we reach num_total_ideas
                improvement_ideas.extend(new_ideas)
                idea_notes.extend(new_notes)
                counter += len(new_ideas)

        # trim the improvement ideas and notes to num_total_ideas
        improvement_ideas = improvement_ideas[:num_total_ideas]
        idea_notes = idea_notes[:num_total_ideas]

        # add the improvement ideas to the bandit learner
        scores = [improvement_prior for _ in improvement_ideas]
        for idea, score, info in zip(improvement_ideas, scores, idea_notes):
            self.mbleaner.add_or_update(idea, None, {'info': info, 'iteration': self.num_evolutions, 'num_implements': 0})

    def generate_improvement_ideas_from_strategy(self, strategy, old_info: dict, num_ideas_per_strategy: int) -> tuple[list[str], list[Any]]:
        '''
        Produces improvement ideas from a strategy and feedback

        Args:
            strategy: strategy to produce improvement ideas from
            old_info: feedback for the strategy
            num_ideas_per_strategy: number of ideas to produce per strategy

        Returns:
            improvement_ideas: improvement ideas
            idea_notes: notes for each idea
        '''
        return self.evolver_dependency.generate_improvement_ideas_from_strategy(old_strategy=strategy, old_info=old_info, num_ideas_per_strategy=num_ideas_per_strategy)
    
    def generate_strategy(self, old_strategy: str, improvement_idea: str, old_info: dict, idea_info: dict) -> tuple[str, dict, bool]:
        return self.evolver_dependency.generate_strategy(old_strategy=old_strategy, improvement_idea=improvement_idea, old_info=old_info, idea_info=idea_info)

    def implement_and_evaluate(self, num_fittest_strategies: int = 1) -> None:
        '''
        Samples 1 idea from the bandit learner, applies it to the top batch_size strategies, and evaluates the results.

        Adds the new strategies to the strategy library.
        Updates the score of the idea based on how much it improved the strategies.

        Args:
            num_fittest_strategies: number of fittest strategies to consider
        '''
        # print('Implementing and evaluating')
        # print('batch size:', self.batch_size)

        # get num_fittest_strategies fittest strategies
        fittest_items = self.get_fittest(num_fittest_strategies)
        
        # propose improvements for the fittest strategies
        proposed_strategies = []
        func_notes = []
        counter = 0 # counts the number of strategies generated
        new_strategy_to_idea_old_score = {} # maps new strategies to (idea, old_strategy, old_score, new_score)

        if len(fittest_items) == 0:
            self.logger.info('No strategies in the library')
            return
        
        while counter < self.batch_size:
            # print(f'Counter: {counter}')
            for strategy, info, prev_score in fittest_items:

                # first sample an improvement idea from the bandit learner
                idea_, idea_notes_, idea_score = self.mbleaner.softmax_sample()

                # generate the new strategy
                new_strategy, new_note, is_generated = self.generate_strategy(strategy, idea_, info, idea_notes_)

                if is_generated:
                    # evaluate new strategy
                    new_score, is_successful = self.evaluate(new_strategy, info)

                    if is_successful:
                    # store the new strategy and its notes
                        new_info = info | {'iteration': self.num_evolutions, 'generation': info['generation'] + 1, 'predecessor_strategy': strategy, 'idea_trace': info['idea_trace'] + tuple([idea_])} | new_note
                        self.add_or_update_strategy(new_strategy, new_info, new_score)
                        proposed_strategies.append(new_strategy)
                        func_notes.append(new_info)

                        new_strategy_to_idea_old_score[new_strategy] = (idea_, strategy, prev_score, new_score)
                
                counter += 1
       
        # update idea score for each strategy
        for i, func in enumerate(proposed_strategies):

            # get the idea, old strategy, old score, and new score
            idea, old_strategy, old_score, new_score = new_strategy_to_idea_old_score[func]
            
            # calculate the improvement score
            improvement_score = new_score - old_score

            # get the idea notes
            idea_notes = self.mbleaner.get_notes_for_arm(idea)

            # increment the number of implementations of the idea by 1
            idea_notes['num_implements'] += 1

            # update the score of the idea
            self.mbleaner.add_or_update(idea, improvement_score, idea_notes)

            self.logger.info(f"Idea {idea} implemented with average improvement score {improvement_score}")
        

    def evolve_once(self) -> None:
        '''
        Evolves the population once
        '''
        self.num_evolutions += 1

        # generate improvement ideas
        self.generate_improvement_ideas(num_fittest_strategies=self.num_fittest_strategies, num_total_ideas=self.num_ideas_per_iteration, num_ideas_per_strategy=self.num_ideas_per_strategy)

        # implement and evaluate
        self.implement_and_evaluate(num_fittest_strategies=self.num_fittest_strategies)

    def get_ideas_df(self, k: int = -1) -> pd.DataFrame:
        '''
        Returns a dataframe of the top k ideas in the bandit learner
        '''
        # store results as a list of dictionaries
        results = []
        top_idea_items = self.mbleaner.get_top_k_items(k)
        for idea, info, score in top_idea_items:
            value = self.mbleaner.get_value_estimate_for_arm(idea)
            to_append = info | {'idea': idea, 'ucb score': score, 'value': value}
            results.append(to_append)

        # convert results to a dataframe
        results_df = pd.DataFrame(results)
        return results_df

