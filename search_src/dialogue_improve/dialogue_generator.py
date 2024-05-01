from search_src.Avalon.baseline_models_Avalon import AvalonState
from search_src.searchlight.utils import AbstractLogged
from search_src.searchlightimprove.llm_utils.llm_api_models import LLMModel
from .prompt_generator import PromptGenerator

class DialogueGenerator(AbstractLogged):
    def __init__(self, llm_model: LLMModel, prompt_generator: PromptGenerator, player, private_information: str, tips: str = ''):
        '''
        Args:
            llm_model: LLM model for dialogue discrimination
            prompt_generator: prompt generator for generating prompts
            player: player index of the agent
            private_information: the private information string given to the player
        '''
        self.llm_model = llm_model # LLM model for dialogue discrimination
        self.prompt_generator = prompt_generator
        self.player = player
        self.private_information = private_information
        self.tips = tips

    def generate_dialogue(self, intended_action, history: str, quest_leader, phase, turn, round, quest_team, team_votes, quest_results,) -> str:
        '''
        Generates dialogue based on the state, intended action and input string

        Args:
            intended_action: the intended action of the agent directly as given by the action planner
            history: input string to generate dialogue
            state_info which contains:
                - quest_leader: quest leader for the current round
                - phase: phase of the game
                - turn: turn of the game
                - round: round of the game
                - quest_team: the team proposed for the quest
                - team_votes: the votes for the team proposal
                - quest_results: the results of the quest
                - good_victory: whether the good side won

        TODO: change intended_action to intended_action_tree in the future
        '''
        prompt = self.prompt_generator.gen_dialogue_generation_prompt(intended_action, history, self.private_information, quest_leader, phase, turn, round, quest_team, team_votes, quest_results, self.tips)
        return self.llm_model.generate(prompt, 1)[0]