from .headers import *

import re

class TreeEvolverDependency(EvolverDependency):
    '''
    Dependency for the BFS evolver
    '''
    @classmethod
    @abstractmethod
    def generate_new_strategy(cls, old_strategy: Any, old_info: dict) -> tuple[Any, dict, bool]:
        '''
        Generates a new strategy based on the old strategy and feedback.
        Feedback will be in the old_info dictionary

        Args:
            old_strategy: old strategy
            old_info: old info

        Returns:
            new_strategy: new strategy
            new_info: notes for the new strategy
            success: whether the generation was successful
        '''
        pass

class StrategistEvolverDependency(EvolverDependency):
    '''
    Dependency for the BFS evolver
    '''

    @classmethod
    @abstractmethod
    def generate_improvement_ideas_from_strategy(cls, old_strategy: Any, num_ideas_per_strategy: int, old_info: dict) -> tuple[list[str], list[Any]]:
        '''
        Generates improvement ideas based on the old strategy and feedback
        Feedback will be in the old_info dictionary

        Args:
            old_strategy: old strategy
            num_ideas_per_strategy: number of ideas to generate per strategy
            old_info: old info for the strategy

        Returns:
            improvement_ideas: list of improvement ideas
            idea_notes: list of notes for each idea
        '''
        pass
    
    @classmethod
    @abstractmethod
    def generate_strategy(cls, old_strategy: Any, improvement_idea: str, old_info: dict, idea_info: dict) -> tuple[str, dict, bool]:
        '''
        Generates a new strategy based on the old strategy and improvement idea

        Args:
            old_strategy: old strategy
            improvement_idea: improvement idea
            old_info: old info for the strategy
            idea_info: info for the improvement idea

        Returns:
            new_strategy: new strategy
            new_info: notes for the new strategy
            success: whether the generation was successful
        '''
        pass

    @classmethod
    def get_idea_generation_signature(cls, num_ideas: int) -> str:
        """
        Returns the signature that ideas should be generated in.

        Below is an example:

        return f'''\n \n Based on the function, feedback, and conclusions you drew, what are {num_ideas} improvements that you can make to the function that you think will have the most impact? Be as specific and concrete as possible, mentioning specific code pieces or helper functions that you can add. If the game has different phases, you can also mention what specific phase the idea applies to. 
        
        Write them out in the following format:
        
        Thoughts: <your thoughts here>

        Idea 1: <your idea here>

        Idea 2: <your idea here>

        ...

        Here's an example of what this might look like for 3 improvement ideas:

        Thoughts: The feedback suggests that the main problem is that function is not taking into account the cards in either player's hand, which leads to inaccurate value estimates.

        Idea 1: I can calculate the difference in the number of cards in each player's hand, increasing the estimated value of players with more cards in their hand.

        Idea 2: I can write a helper function that compares the sum of the values of the cards in each player's hand, increasing the estimated value of players with higher total card values.

        Idea 3: During the draw phase, I can also calculate the expected values of the cards that the players will draw, increasing the estimated value of players with higher expected card values.
        '''
        """
        raise NotImplementedError

    @staticmethod
    def parse_improvement_ideas(string: str, num_ideas: int) -> list[str]:
        # Regular expression to find patterns of "Idea X: <idea here>"
        idea_pattern = re.compile(r"Idea \d+: (.+)")
        
        # Find all matches of the pattern in the input string
        matches = idea_pattern.findall(string)
        
        # Return the first num_ideas ideas from the matches
        return matches[:num_ideas]
