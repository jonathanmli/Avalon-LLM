from .headers import *
# from .prompts.improvement_prompts import gen_single_state_example_feedback, gen_execution_error_feedback_2
from .prompts.prompt_generators import PromptGenerator, StrategyPromptGenerator
import numpy as np

class HeuristicsAnalyzer(FeedbackAnalyzer):
    '''
    Analyzes numerical feedback and translates into natural language feedback
    '''
    def __init__(self, prompt_generator: PromptGenerator, num_samples: int = 6, rng: np.random.Generator = np.random.default_rng(), search_guided_sampling: bool = False):
        super().__init__()
        self.num_samples = num_samples
        self.rng = rng
        self.gen_state_description = prompt_generator.gen_state_description
        self.prompt_generator = prompt_generator
        self.search_guided_sampling = search_guided_sampling

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

            if not self.search_guided_sampling:
                feedback = self.sample_feedback(trajectory_data)
            else:
                feedback = self.get_top_states_by_search_discrepancy(trajectory_data)
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
                    feedback += self.prompt_generator.gen_outcome_single_state_example_feedback(i, self.gen_state_description(episode_data['trajectory'][state_index][2]), 
                                                                episode_data['heuristics_score_trajectory'][state_index], 
                                                                episode_data['heuristics_trajectory'][state_index], 
                                                                episode_data['search_trajectory'][state_index], 
                                                                episode_data['score_trajectory'][state_index])
        return feedback

    def get_top_states_by_search_discrepancy(self, trajectory_data: list[dict]) -> str:
        '''
        Samples feedback from trajectory data, selecting the top states across all episodes with the highest L2 norm differences 
        between search estimates and heuristic scores for each player, excluding states with N/A values.

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
        all_state_info = []
        for episode_data in trajectory_data:
            if len(episode_data['trajectory']) > 1:
                # Filter out any N/A states (empty dictionaries) from the lists
                filtered_search_trajectory = [s for s in episode_data['search_trajectory'][1:] if s]
                filtered_heuristics_score_trajectory = [h for h in episode_data['heuristics_score_trajectory'][1:] if h]

                # Ensure both lists remain aligned by filtering based on valid indices
                valid_indices = [i for i, (s, h) in enumerate(zip(episode_data['search_trajectory'][1:], episode_data['heuristics_score_trajectory'][1:]), start=1) if s and h]
                differences = HeuristicsAnalyzer.calculate_l2_norm_differences(filtered_search_trajectory, filtered_heuristics_score_trajectory)

                for i, diff in zip(valid_indices, differences):
                    # Collect state information along with their differences
                    all_state_info.append((diff, i, episode_data))

        # Sort all collected states by the differences, descending order
        all_state_info.sort(reverse=True, key=lambda x: x[0])

        # Select the top num_samples states
        top_states = all_state_info[:self.num_samples]

        # Generate feedback for the selected states
        feedback = ''
        for _, state_index, episode_data in top_states:
            feedback += self.prompt_generator.gen_single_state_example_feedback(state_index, self.gen_state_description(episode_data['trajectory'][state_index][2]), 
                                                    episode_data['heuristics_score_trajectory'][state_index], 
                                                    episode_data['heuristics_trajectory'][state_index], 
                                                    episode_data['search_trajectory'][state_index], 
                                                    episode_data['score_trajectory'][state_index])

        return feedback
    
    @staticmethod
    def calculate_l2_norm_differences(list1: list[dict], list2: list[dict]) -> list:
        '''
        Calculates the L2 norm of the difference vectors between two lists of dictionaries. Each dictionary 
        in the lists corresponds to values for players, identified by the same keys.

        Args:
            list1: A list of dictionaries, where each dictionary contains player values for one state.
            list2: A list of dictionaries, with the same structure as list1.

        Returns:
            differences: A list of L2 norms of the difference vectors for each corresponding pair of dictionaries.
        '''
        differences = []
        for dict1, dict2 in zip(list1, list2):
            if dict1.keys() != dict2.keys():
                raise ValueError("Dictionaries must have the same set of keys")
            # Convert dictionary values to arrays and compute the L2 norm of the difference
            diff_vector = np.array(list(dict1.values())) - np.array(list(dict2.values()))
            norm = np.linalg.norm(diff_vector)
            differences.append(norm)
        return differences


class DialogueAnalyzer(FeedbackAnalyzer):

    def __init__(self, prompt_generator: StrategyPromptGenerator):
        super().__init__()
        self.prompt_generator = prompt_generator

    def translate(self, data: dict) -> str:
        '''
        Analyzes and translates feedback into an natural language feedback

        Args:
            data: feedback data for a single agent, which looks like this:
            {"dialogues": generated_dialogues, "feedbacks": feedbacks, "evaluated_players": evaluated_players, "discussion_history": discussion_history}
        '''
        outputs = []
        for i, (dialogue, feedback, player, thought) in enumerate(zip(data['dialogues'], data['feedbacks'], data['evaluated_players'], data["thoughts"])):
            output = self.prompt_generator.gen_strategy_feedback(i, dialogue, feedback, player, thought)
            outputs.append(output)

        return '\n'.join(outputs)
    
class OutcomeDialogueAnalyzer(FeedbackAnalyzer):

    def __init__(self, prompt_generator: StrategyPromptGenerator):
        super().__init__()
        self.prompt_generator = prompt_generator

    def translate(self, data: dict) -> str:
        '''
        Analyzes and translates feedback into an natural language feedback

        Args:
            data: feedback data for a single agent, which looks like this:
            {"dialogues": generated_dialogues, "feedbacks": feedbacks, "evaluated_players": evaluated_players, "discussion_history": discussion_history}
        '''
        outputs = []
        for i, (dialogue, feedback, player, thought, score) in enumerate(zip(data['dialogues'], data['feedbacks'], data['evaluated_players'], data["thoughts"], data["scores"])):
            output = self.prompt_generator.gen_outcome_strategy_feedback(i, dialogue, feedback, player, thought, score)
            outputs.append(output)

        return '\n'.join(outputs)
    
class LLMCriticAnalyzer(FeedbackAnalyzer):

    def __init__(self):
        super().__init__()

    def translate(self, data: dict) -> str:
        '''
        Analyzes and translates feedback into an natural language feedback

        Args:
            data: feedback data for a single agent, which looks like this:
            {"explanation": explanation, "score": score}
        '''
        output = f"Your strategy received a score of {data['score']} from the critic. Here is the explanation: {data['explanation']}"
        return output


