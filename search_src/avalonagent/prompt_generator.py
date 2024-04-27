from ..Avalon.prompts import GAME_RULES

SYS_PROMPT = "You are a player in the game of Resistance: Avalon (Avalon)."

SUMMARIZATION = "Summarize the dialogue history. Try to keep all useful information that would be helpful towards winning the game, including your observations of the other players motives."

GENERATION = "Given the discussion history, produce dialogue that would help you achieve your intent. Dialogue aimed at persuasion should be presented in a convincing manner, supported by ethos, logos, and pathos. Note that dialogue will be seen by all players in the game."


class PromptGenerator:

    def __init__(self, private_information, rules=GAME_RULES, sys_prompt = SYS_PROMPT):
        self.rules = rules
        self.sys_prompt = sys_prompt
        self.private_information = private_information # i.e. role, private knowledge

    def gen_summarization_prompt(self):
        return self.sys_prompt + self.rules + self.private_information + SUMMARIZATION
    
    def generate_belief_discrimination_prompt_for_player(self, history: str, player, is_evil=False):
        if not is_evil:
            discrim_str = f"Based on the discussions this round, do you think the probability of player {player} being Evil (1) increased significantly (2) increased slightly (3) stayed the same (4) decreased slightly or (5) decreased significantly and why?"
        else:
            discrim_str = f"Based on the discussions this round, do you think the probability of player {player} being Merlin (1) increased significantly (2) increased slightly (3) stayed the same (4) decreased slightly or (5) decreased significantly and why?"
        return self.sys_prompt + self.rules + self.private_information + history + discrim_str
    
    def generate_belief_discrimination_prompt(self, history: str, players: set, is_evil=False):
        if not is_evil:
            discrim_str = f"Based on the discussions this round, for each player in players {players}, do you think the probability of the player being Evil (1) increased significantly (2) increased slightly (3) stayed the same (4) decreased slightly or (5) decreased significantly and why? End with a dictionary of player to your multiple choice answer, i.e. {{0: (1, 'increased significantly'), ..., 4: (3, 'stayed the same')}}"
        else:
            discrim_str = f"Based on the discussions this round, for each player in players {players}, do you think the probability of the player being Merlin (1) increased significantly (2) increased slightly (3) stayed the same (4) decreased slightly or (5) decreased significantly and why? End with a dictionary of player to your multiple choice answer, i.e. {{0: (1, 'increased significantly'), ..., 4: (3, 'stayed the same')}}"
        return self.sys_prompt + self.rules + self.private_information + history + discrim_str

    def generate_intent_discrimination_prompt_for_player(self, history: str, player):
        discrim_str = f"Based on the discussions this round, what action do you think {player} will likely take next and why?"

        return self.sys_prompt + self.rules + self.private_information + history + discrim_str
    
    def gen_dialogue_generation_prompt(self, state, intended_action, history: str):
        return self.sys_prompt + self.rules + self.private_information + history + GENERATION
