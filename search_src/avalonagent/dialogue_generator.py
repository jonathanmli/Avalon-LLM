from Avalon.baseline_models_Avalon import AvalonState
from searchlight.utils import AbstractLogged

class DialogueGenerator(AbstractLogged):
    def __init__(self, llm_model):
        self.llm_model = llm_model # LLM model for dialogue discrimination

    def generate_dialogue(self, state: AvalonState, intended_action, history: str) -> str:
        '''
        Generates dialogue based on the state, intended action and input string
        '''
        raise NotImplementedError