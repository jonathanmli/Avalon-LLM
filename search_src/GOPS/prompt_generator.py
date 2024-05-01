from search_src.searchlightimprove.prompts.prompt_generators import PromptGenerator
from search_src.searchlightimprove.prompts.improvement_prompts import SYS_PROMPT,SYS_PROMPT_POLICY
from .prompts import GOPS_RULES, VALUE_FUNCTION_SIGNATURE, POLICY_FUNCTION_SIGNATURE 
from .baseline_models_GOPS import GOPSState2

class GOPSPromptGenerator(PromptGenerator):
    '''
    Generates prompts for GOPS action self improvement 
    '''
    def __init__(self, function_signature: str, sys_prompt: str):
        super().__init__(environment_rules=GOPS_RULES, function_signature=function_signature, sys_prompt=sys_prompt)
    
    @staticmethod
    def gen_state_description(state: GOPSState2):
        state_description = f'''The current state of the game is as follows:
        - The score cards that have been revealed are: {state.prize_cards}
        - The cards that player 0 has played are: {state.player_cards}
        - The cards that player 1 has played are: {state.opponent_cards}
        - Player 0's score so far is: {state.get_scores()[0]}
        - Player 1's score so far is: {state.get_scores()[1]}
        - The score cards left in the deck are: {state.get_score_deck()}
        - The cards left in player 0's hand are: {state.get_player_hand()}
        - The cards left in player 1's hand are: {state.get_opponent_hand()}
        '''
        return state_description
    
class GOPSValuePromptGenerator(GOPSPromptGenerator):

    def __init__(self):
        super().__init__(function_signature=VALUE_FUNCTION_SIGNATURE, sys_prompt=SYS_PROMPT)

class GOPSPolicyPromptGenerator(GOPSPromptGenerator):

    def __init__(self):
        super().__init__(function_signature=POLICY_FUNCTION_SIGNATURE, sys_prompt=SYS_PROMPT_POLICY)