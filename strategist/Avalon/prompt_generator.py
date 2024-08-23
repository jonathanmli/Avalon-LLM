from strategist.searchlightimprove.prompts.prompt_generators import PromptGenerator
from strategist.searchlightimprove.prompts.improvement_prompts import SYS_PROMPT,SYS_PROMPT_POLICY
from .prompts import GAME_RULES, HEURISTICS_FUNCTION_SIGNATURE
from .baseline_models_Avalon import AvalonState

class AvalonPromptGenerator(PromptGenerator):
    '''
    Generates prompts for Avalon action self improvement
    '''
    def __init__(self, function_signature: str, sys_prompt: str):
        super().__init__(environment_rules=GAME_RULES, function_signature=function_signature, sys_prompt=sys_prompt, seed_heuristic_thought_prompt=1)
    
    @staticmethod
    def gen_state_description(state: AvalonState, with_hidden_info=True):
        if with_hidden_info:
            hidden_info = f'''- The roles of the players in order are {state.get_roles_in_str_list()} with sides {state.get_is_good()} (True for Good, False for Evil)
            '''
        else:
            hidden_info = ''

        if state.phase == 0:
            phase_description = f'''- The current phase of the game is the team selection phase
            - The current leader is player {state.quest_leader}
            '''
        elif state.phase == 1:
            phase_description = f'''- The current phase of the game is the team voting phase
            - The current leader is player {state.quest_leader} who selected {state.quest_team} as the quest team
            '''
        elif state.phase == 2:
            phase_description = f'''- The current phase of the game is the quest voting phase
            - The team {state.quest_team} was approved with {state.quest_votes.count(True)} votes for and {state.quest_votes.count(False)} votes against
            '''
        elif state.phase == 3:
            phase_description = f'''- The current phase of the game is the assassination phase
            - The assassin is player {state.get_assassin()}
            '''
        state_description = f'''The current state of the game is as follows:
        - The number of players in the game is: {state.config.num_players}
        - This is the quest number {state.turn} which requires {state.config.num_players_for_quest[state.turn]} players and {state.config.num_fails_for_quest[state.turn]} fails to fail 
        - This is the {state.round} round of discussion
        - The previous results for the quest were {state.quest_results} (True for Success, False for Fail)
        ''' + hidden_info + phase_description
        return state_description
    
class AvalonValuePromptGenerator(AvalonPromptGenerator):

    def __init__(self):
        super().__init__(function_signature=HEURISTICS_FUNCTION_SIGNATURE, sys_prompt=SYS_PROMPT)

# class AvalonPolicyPromptGenerator(AvalonPromptGenerator):

#     def __init__(self):
#         super().__init__(function_signature=POLICY_FUNCTION_SIGNATURE, sys_prompt=SYS_PROMPT_POLICY)