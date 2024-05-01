from .llm_utils.llm_api_models import LLMModel, GPT35Multi
import logging
import pandas as pd
import plotly.express as px
import datetime
from abc import ABC, abstractmethod
from search_src.searchlight.utils import AbstractLogged

# from .prompts.improvement_prompts import gen_execution_error_feedback
from .prompts.prompt_generators import PromptGenerator

from typing import Any, Optional, Callable

class ImprovementProposer(ABC):
    '''
    Abstract class for an improvement proposer, which takes as input a base string and proposes a collection of improved strings
    '''
    def __init__(self):
        # Create a logger for each instance with the name of the class
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def propose(self, base: str, abstract: str = '', feedback: str = '') -> tuple[list[str], list[str]]:
        '''
        Proposes improvements to a single base string (usually a function as a string)

        Args:
            base: base string to improve
            abstract: abstract representation of the base string
            feedback: any notes about the prompt (e.g. context, specific suggestions)
                which can include the following:
                - notes['heuristic']: feedback from the heuristic function
                - notes['execution_error']: feedback from the execution error function
                - notes['trajectory']: trajectory of simulated games

        Returns:
            proposed_functions: collection of proposed functions
            proposed_abstracts: collection of abstract representations of the proposed functions
        '''
        pass
    
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
    
    def evaluate_with_benchmark(self, objects: list[Any]) -> tuple[list[float], list, dict[str, float]]:
        '''
        Evaluates a collection of functions using a benchmark

        Args:
            functions: collection of functions to evaluate

        Returns:
            function_scores: list of scores for each function
            function_notes: list of notes for each function
            benchmark_scores: dictionary of benchmark scores from name to score
        '''
        function_scores, function_notes = self.evaluate(objects)
        return function_scores, function_notes, dict()
    
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
    def __init__(self):
        pass

    @abstractmethod
    def evolve(self, num_cycles: int):
        '''
        Evolves the functions for a certain number of cycles
        '''
        pass
        
