# from .llm_utils.llm_api_models import LLMModel, GPT35Multi
# import logging
# import pandas as pd
# import plotly.express as px
# import datetime
from abc import ABC, abstractmethod
from searchlight.utils import AbstractLogged

# from .prompts.improvement_prompts import gen_execution_error_feedback
# from .prompts.prompt_generators import PromptGenerator

from typing import Any, Optional, Callable
    
class Evaluator(AbstractLogged):
    '''
    Abstract class for an evaluator, which evaluates a collection of functions
    '''
    def __init__(self):
        super().__init__()

    @abstractmethod
    def evaluate(self, objects: list[Any]) -> tuple[list[float],list[dict]]:
        '''
        Evaluates a collection of functions

        Args:
            functions: collection of functions to evaluate

        Returns:
            scores: scores of the functions
            notes: notes for each function
        '''
        pass
    
class FeedbackAnalyzer(AbstractLogged):
    '''
    Abstract class for analyzing numerical feedback and translating into natural language feedback
    '''
    def __init__(self):
        super().__init__()

    @abstractmethod
    def translate(self, data) -> str:
        '''
        Analyzes and translates feedback into an abstract representation

        Args:
            data: feedback data

        Returns:
            feedback_str: natural language feedback
        '''
        pass
    
class Evolver(AbstractLogged):
    '''
    Abstract class for evolving functions
    '''

    @abstractmethod
    def evolve(self, num_cycles: int):
        '''
        Evolves the functions for a certain number of cycles
        '''
        pass

class EvolverDependency(AbstractLogged):
    '''
    Abstract class that must be defined for evolution process
    '''

    @classmethod
    @abstractmethod
    def evaluate(cls, strategy: Any, strategy_info: dict) -> tuple[float, bool]:
        '''
        Evaluates an strategy. Modifies the strategy_info in place

        Args:
            strategy: strategy to evaluate
            strategy_info: info for the strategy

        Returns:
            score: score of the strategy
            success: whether the evaluation was successful
        '''
        pass



class Evolver2(AbstractLogged):
    '''
    Abstract class for evolving strategies
    '''

    def __init__(self, evolver_dependency: EvolverDependency):
        super().__init__()
        self.evolver_dependency = evolver_dependency

    def evolve(self, num_cycles: int) -> None:
        '''
        Evolves the strategies for a certain number of cycles
        '''
        self._evolve(num_cycles)
        
    @abstractmethod
    def _evolve(self, num_cycles: int) -> None:
        '''
        Evolves the strategies for a certain number of cycles
        '''
        pass

    @abstractmethod
    def get_best_strategy(self) -> tuple[Any, float, dict]:
        '''
        Returns the best strategy

        Returns:
            best_strategy: best strategy
            best_score: best score
            best_info: best info
        '''
        pass

class StrategyLibrary(AbstractLogged):
    '''
    Abstract class for storing and retrieving strategies to improve upon
    '''
    @abstractmethod
    def add_or_update_strategy(self, strategy: Any, notes: dict, score: float):
        '''
        Adds or updates a strategy in the strategies dictionary

        Args:
            strategy: strategy to add or update
            notes: notes for the strategy
            score: score of the strategy
        '''
        pass

    @abstractmethod
    def select_strategies(self, k: int = 1) -> list[tuple[Any, dict, float]]:
        '''
        Selects the k strategies to improve upon

        Args:
            k: number of strategies to select. If k = -1, select all strategies

        Returns:
            items: items of the form (strategy, info, priority)
        '''
        pass

    @abstractmethod
    def get_best_strategy(self) -> tuple[Any, dict, float]:
        '''
        Returns the best strategy

        Returns:
            best_strategy: best strategy
            best_info: best info
            best_score: best score
        '''
        pass



        
