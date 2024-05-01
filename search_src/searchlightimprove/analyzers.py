from .headers import *
# from .prompts.improvement_prompts import gen_single_state_example_feedback, gen_execution_error_feedback_2
from .prompts.prompt_generators import PromptGenerator
import numpy as np

class HeuristicsAnalyzer(FeedbackAnalyzer):
    '''
    Analyzes numerical feedback and translates into natural language feedback
    '''
    def __init__(self, prompt_generator: PromptGenerator, num_samples: int = 6, rng: np.random.Generator = np.random.default_rng(), ):
        super().__init__()
        self.num_samples = num_samples
        self.rng = rng
        self.gen_state_description = prompt_generator.gen_state_description
        self.prompt_generator = prompt_generator

    def translate(self, data: dict) -> str:
        '''
        Analyzes and translates feedback into an natural language feedback

        Args:
            data: feedback data for a single agent, which has the following keys:
                - 'execution_error': execution error feedback. if there are execution errors, this might be the only feedback
                - 'trajectory_data': trajectory data, which is a list of dictionaries, one for each episode, with the following keys:
                    - 'trajectory': trajectory of the game
                    - 'heuristics_trajectory': trajectory of heuristic feedback
                    - 'search_trajectory': trajectory of search estimates
                    - 'score_trajectory': trajectory of scores 

        Returns:
            feedback: natural language feedback
        '''
        if 'execution_error' in data:
            feedback = self.prompt_generator.gen_execution_error_feedback_2(data['execution_error'])
            return feedback
        else:
            trajectory_data: list[dict] = data['trajectory_data']
            feedback = self.sample_feedback(trajectory_data)
            return feedback

    def sample_feedback(self, trajectory_data: list[dict]) -> str:
        '''
        Samples feedback from a trajectory data

        Args:
            trajectory_data: trajectory data, which is a list of dictionaries, one for each episode, with the following keys:
                - 'trajectory': trajectory of the game
                - 'heuristics_score_trajectory': trajectory of heuristic scores
                - 'heuristics_trajectory': trajectory of heuristic feedback
                - 'search_trajectory': trajectory of search estimates
                - 'score_trajectory': trajectory of scores 

        Returns:
            feedback: natural language feedback
        '''
        feedback = ''
        for i in range(self.num_samples):
            if len(trajectory_data) > 0:
                # first sample a random episode
                episode_index = self.rng.integers(len(trajectory_data))
                episode_data = trajectory_data[episode_index]

                # then sample a random state from the episode. make sure not to sample the first state
                # NOTE: if the episode is empty, we skip it. we might those produce less feedback than num_samples
                if len(episode_data['trajectory']) > 1:

                    # log the lengths of the data for debugging
                    self.logger.info(f'lengths: trajectory {len(episode_data["trajectory"])} heuristics_score_trajectory {len(episode_data["heuristics_score_trajectory"])} heuristics_trajectory {len(episode_data["heuristics_trajectory"])} search_trajectory {len(episode_data["search_trajectory"])} score_trajectory {len(episode_data["score_trajectory"])}')

                    assert(len(episode_data['trajectory']) == len(episode_data['heuristics_score_trajectory']) == len(episode_data['heuristics_trajectory']) == len(episode_data['search_trajectory']) == len(episode_data['score_trajectory']))
                    state_index = self.rng.integers(1, len(episode_data['trajectory']))
                    feedback += self.prompt_generator.gen_single_state_example_feedback(i, self.gen_state_description(episode_data['trajectory'][state_index][2]), 
                                                                episode_data['heuristics_score_trajectory'][state_index], 
                                                                episode_data['heuristics_trajectory'][state_index], 
                                                                episode_data['search_trajectory'][state_index], 
                                                                episode_data['score_trajectory'][state_index])
        return feedback
