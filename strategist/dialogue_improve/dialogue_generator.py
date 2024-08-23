# from ..Avalon.baseline_models_Avalon import AvalonState
from strategist.searchlight.utils import AbstractLogged
from strategist.searchlightimprove.llm_utils.llm_api_models import LLMModel
from .prompt_generator import PromptGenerator

class DialogueGenerator(AbstractLogged):
    def __init__(self, llm_model: LLMModel, prompt_generator: PromptGenerator,):
        '''
        Args:
            llm_model: LLM model for dialogue discrimination
            prompt_generator: prompt generator for generating prompts
            player: player index of the agent
            private_information: the private information string given to the player
        '''
        super().__init__()
        self.llm_model = llm_model # LLM model for dialogue discrimination
        self.prompt_generator = prompt_generator



    def generate_dialogue(self, intended_action, phase: int, history: str, tips: str) -> tuple[str, dict]:
        '''
        Generates dialogue based on the state, intended action and input string

        Args:
            intended_action: the intended action of the agent directly as given by the action planner
            history: history of the game so far, including dialogue and state, in string form

        TODO: change intended_action to intended_action_tree in the future
        '''
        prompt = self.prompt_generator.gen_dialogue_generation_thought_prompt(intended_action, phase, history, tips)
        response = self.llm_model.generate(prompt, 1)[0]
        # print("Generator thought prompt: ", prompt)
        
        prompt = self.prompt_generator.gen_dialogue_generation_action_prompt(intended_action, phase, history, response)
        response2 = self.llm_model.generate(prompt, 1)[0]
        # print("Generator action prompt: ", prompt)
        # self.logger.info(f'Generated dialogue process: {prompt + response2}')
        return response2, {'thought': response}