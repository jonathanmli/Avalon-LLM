from ..Avalon.prompts import GAME_RULES
from ..Avalon.baseline_models_Avalon import AvalonState, AvalonBasicConfig
from ..Avalon.prompt_generator import AvalonPromptGenerator
from search_src.searchlight.utils import AbstractLogged

SYS_PROMPT = "You are a player in the game of Resistance: Avalon (Avalon). "

SUMMARIZATION = "Summarize the dialogue history. Try to keep all useful information that would be helpful towards winning the game, including your observations of the other players motives."

GENERATION = "\n \n Given the discussion history, current state, and your intended action, produce dialogue that would help you achieve your intent. Dialogue aimed at persuasion should be presented in a convincing manner, supported by ethos, logos, and pathos. Note that dialogue will be seen by all players in the game, hence your generated response should not include your role or any private information. \n You dialogue: \n"

GENERATION_THOUGHT_GUIDE = "\n \n Given the discussion history, current state, and your intended action, answer the questionaire to the best of your ability. The answers should be based on the information you have and your own reasoning. \n \n "

# GENERATION_ACTION = "\n \n Given the discussion history, current state, your intended action, and the questionarie you just answered, produce dialogue that would help you achieve your intent. Dialogue aimed at persuasion should be presented in a convincing manner, supported by ethos, logos, and pathos. Note that dialogue will be seen by all players in the game, hence your generated response should not include your role or any private information. \n Your dialogue: \n "

# GENERATION_ACTION = "\n \n Assemble your responses to the questionaire you just answered into a speech that would help you achieve your intent. Note that dialogue will be seen by all players in the game, hence your generated response should not include your role or any private information. \n Your speech: \n "

GENERATION_ACTION = "\n \n Assemble your responses to the questionaire you just answered into a speech that would help you achieve your intent. Also include responses to dialogue from other players this round if appropriate. Note that dialogue will be seen by all players in the game, so you should not reveal your role (identity). \n Your speech: \n "

class PromptGenerator(AbstractLogged):

    def __init__(self, config: AvalonBasicConfig, rules=GAME_RULES, sys_prompt = SYS_PROMPT, ):
        self.rules = rules
        self.sys_prompt = sys_prompt
        self.config = config
        # self.private_information = private_information # i.e. role, private knowledge about other players
        super().__init__()

    def gen_summarization_prompt(self, private_information: str):
        return self.sys_prompt + self.rules + private_information + SUMMARIZATION
    
    def generate_belief_discrimination_prompt_for_player(self, history: str, player, private_information: str, is_evil=False):
        if not is_evil:
            discrim_str = f"Based on the discussions this round, do you think the probability of player {player} being Evil (2) increased significantly (1) increased slightly (0) stayed the same (-1) decreased slightly or (-2) decreased significantly and why?"
        else:
            discrim_str = f"Based on the discussions this round, do you think the probability of player {player} being Merlin (1) increased significantly (2) increased significantly (1) increased slightly (0) stayed the same (-1) decreased slightly or (-2) decreased significantly and why?"
        return self.sys_prompt + self.rules + private_information + history + discrim_str
    
    def generate_pgood_belief_discrimination_prompt(self, history: str, players: set):
        discrim_str = f"""\n \n Based on the discussions this round, for each player in players {players}, do you think the probability of the player being Evil (2) increased significantly (1) increased slightly (0) stayed the same (-1) decreased slightly or (-2) decreased significantly and why? Begin with your reasoning and end with a dictionary of player to your multiple choice answer, i.e. {{0: (2, 'increased significantly'), ..., 4: (0, 'stayed the same')}}. Make sure you include quotes around the dictionary values that are strings.
        
        Here's an example:

        Thought: 
        Player 4's speech seems overly enthusiastic and focused on unity and cooperation, which could be a tactic to divert attention from their true identity as an Evil player. This raises suspicion and increases the probability of Player 4 being Evil. On the other hand, Player 3's response as a Servant of Arthur is more focused on the success of the Quests and unity, which aligns with the goals of Good and decreases the probability of Player 3 being Evil.
        
        Dictionary: 
        {{0: (0, 'stayed the same'), 1: (0, 'stayed the same'), 2: (2, 'increased significantly'), 3: (-2, 'decreased significantly'), 4: (0, 'stayed the same')}} \n \n"""

        return self.sys_prompt + self.rules + history + discrim_str
    
    def generate_pmerlin_belief_discrimination_prompt(self, history: str, players: set,):
        discrim_str = f"""\n \n Based on the discussions this round, for each player in players {players}, do you think the probability of the player being Merlin (2) increased significantly (1) increased slightly (0) stayed the same (-1) decreased slightly or (-2) decreased significantly and why? Begin with your reasoning and end with a dictionary of player to your multiple choice answer, i.e. {{0: (2, 'increased significantly'), ..., 4: (0, 'stayed the same')}}. Make sure you include quotes around the dictionary values that are strings.
        
        Here's an example:

        Thought: 
        Player 2 seems to suspect the Evil players (Player 3 and Player 4) without any evidence and is trying to steer the conversation towards them. This aligns with the behavior of Merlin, who knows the identity of the Evil players and is trying to subtly guide the Good players towards the correct decisions. This increases the probability of Player 2 being Merlin. On the other hand, Player 4's speech seems generally clueless and lacks any strategic insight, which decreases the probability of Player 4 being Merlin.
        
        Dictionary: 
        {{0: (0, 'stayed the same'), 1: (0, 'stayed the same'), 2: (2, 'increased significantly'), 3: (-2, 'decreased significantly'), 4: (0, 'stayed the same')}}\n \n """
        return self.sys_prompt + self.rules + history + discrim_str
    
    def generate_parsing_error_prompt(self, prev_prompt: str, response: str, error: str):
        return prev_prompt + response + f"Error parsing the response you generated: {error}.\n Please try again. \n \n"

    def generate_intent_discrimination_prompt_for_player(self, history: str, player, private_information: str):
        discrim_str = f"Based on the discussions this round, what action do you think {player} will likely take next and why?"

        return self.sys_prompt + self.rules + private_information + history + discrim_str
    
    def gen_dialogue_generation_prompt(self, intended_action, history: str, private_information: str, quest_leader, phase, turn, round, quest_team, team_votes, quest_votes, tips: str = ''):
        state_description = self.gen_state_description_from_info(quest_leader, phase, turn, round, quest_team, team_votes, quest_votes,)
        intended_action_description = PromptGenerator.gen_intended_action_description(intended_action, phase= phase)
        return self.sys_prompt + self.rules + private_information + history + state_description + intended_action_description + tips + GENERATION
    
    def gen_dialogue_generation_thought_prompt(self, intended_action, phase: int, history: str, tips: str):
        intended_action_description = PromptGenerator.gen_intended_action_description(intended_action, phase= phase)
        return self.sys_prompt + self.rules + history + intended_action_description + tips + GENERATION_THOUGHT_GUIDE
    
    # def gen_dialogue_generation_action_prompt(self, prev: str):
    #     return prev + GENERATION_ACTION

    def gen_dialogue_generation_action_prompt(self, intended_action, phase: int, history: str, thoughts):
        intended_action_description = PromptGenerator.gen_intended_action_description(intended_action, phase=phase)
        return self.sys_prompt + history + intended_action_description + "\n \n These are your previous thoughts: \n" + thoughts + GENERATION_ACTION
    
    @staticmethod
    def gen_dialogue_description(dialogue: list[str], speakers: list[int]) -> str:
        # should return Player 1: "Dialogue 1" \n Player 2: "Dialogue 2" \n ...
        dialogue_str = "\n This discussion round players have made the following statements so far: \n \n"
        for i, speaker in enumerate(speakers):
            dialogue_str += f"Player {speaker}: \"{dialogue[i]}\"\n"
        return dialogue_str
    
    @staticmethod
    def gen_summary_preamble(summary: str) -> str:
        print(f"Summary: {summary}")
        return "\n \n Here is a summary of previous rounds of discussion so far:\n \n " + summary
    
    @staticmethod
    def gen_state_description(state: AvalonState) -> str:
        return AvalonPromptGenerator.gen_state_description(state, with_hidden_info=False)
    
    def gen_state_description_from_info(self, quest_leader, phase, turn, round, quest_team, team_votes, quest_results,):
        if phase == 0:
            phase_description = f'''- The current phase of the game is the team selection phase
            - The current leader is player {quest_leader}
            '''
        elif phase == 1:
            phase_description = f'''- The current phase of the game is the team voting phase
            - The current leader is player {quest_leader} who selected {quest_team} as the quest team
            '''
        elif phase == 2:
            phase_description = f'''- The current phase of the game is the quest voting phase
            - The team {quest_team} was approved with {team_votes.count(True)} votes for and {team_votes.count(False)} votes against
            '''
        elif phase == 3:
            phase_description = f'''- The current phase of the game is the assassination phase
            '''
        else:
            raise ValueError(f"Invalid phase {phase}")
        state_description = f'''\n The current state of the game is as follows:
        - The number of players in the game is: {self.config.num_players}
        - This is the quest number {turn} which requires {self.config.num_players_for_quest[turn]} players and {self.config.num_fails_for_quest[turn]} fails to fail 
        - This is the {round} round of discussion
        - The previous results for the quest were {quest_results} (True for Success, False for Fail)
        ''' +  phase_description
        return state_description

    VOTE_ACTION_TO_STR = {True: "approve", False: "reject"}
    QUEST_ACTION_TO_STR = {True: "succeed", False: "fail"}

    @staticmethod
    def gen_intended_action_description(intended_action, phase) -> str:
        if phase == 0: # proposal phase
            intended_action_str = f"You would like the following team to be approved:  {intended_action}\n"
        elif phase == 1: # voting phase
            intended_action_str = f"You would like to vote {PromptGenerator.VOTE_ACTION_TO_STR[intended_action]} on the team proposal.\n"
        elif phase == 2: # quest phase
            intended_action_str = f"You would like to {PromptGenerator.QUEST_ACTION_TO_STR[intended_action]} the quest.\n"
        elif phase == 3: # assassin phase
            intended_action_str = f"You would like to assassinate player {intended_action}.\n"
        else:
            raise ValueError(f"Invalid phase {phase}")
        
        return intended_action_str
    
    # state_info: (quest_leader, phase, turn, round, done, quest_team, team_votes, quest_votes, quest_results, good_victory)