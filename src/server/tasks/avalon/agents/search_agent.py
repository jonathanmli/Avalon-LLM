from .llm_with_discussion import LLMAgentWithDiscussion # this part is not correct yet
from ..engine import AvalonBasicConfig
from ..wrapper import AvalonSessionWrapper, Session
from search_src.Avalon.baseline_models_Avalon import AvalonState
from search_src.dialogue_improve.action_planner import ActionPlanner
from search_src.dialogue_improve.dialogue_generator import DialogueGenerator
from search_src.dialogue_improve.dialogue_discrimination import DialogueDiscriminator
from search_src.dialogue_improve.prompt_generator import PromptGenerator
from search_src.searchlightimprove.llm_utils.llm_api_models import GPT35Multi
from ..prompts import *

class SearchlightLLMAgentWithDiscussion(LLMAgentWithDiscussion):
    def __init__(self, name: str, num_players: int, id: int, role: int, role_name: str, config:AvalonBasicConfig, session: AvalonSessionWrapper=None, side=None, seed=None, func_str=None, **kwargs):
        # we want to keep track of the current game state here somehow. this should include both the current action history and dialogue history (or summary of dialogue history)
        # we will use modules from the dialogue_discrimination.py and dialogue_generator.py to update the game state and generate dialogue
        # and some action intent prediction module to predict the action intent of the agent
        super().__init__(
            name=name,
            num_players=num_players,
            id=id,
            role=role,
            role_name=role_name,
            config=config,
            session=session,
            side=side,
            seed=seed,
            **kwargs
        )
        # TODO: instantiate the followings after initialize_game_info
        self.action_planner = ActionPlanner(
            config=self.config,
            func_str=func_str,
            player=self.id,
            player_role=self.role,
            known_sides=self.side
        )
        self.prompt_generator = PromptGenerator()


    async def initialize_game_info(self, player_list, **kwargs) -> None:
        """Initiliaze the game info for the agent, which includes game introduction, role, and reveal information for different roles."""
        # Introduction Prompt
        verbal_side = ["Evil", "Good"]
        intro_prompt = INTRODUCTION
        intro_prompt += '\n'
        content_prompt = intro_prompt + INFO_ROLE.format(self.num_players, self.num_good, int(self.merlin), self.num_good - int(self.merlin) - int(self.percival), self.num_evil, self.num_evil - int(self.morgana) - int(self.mordred) - int(self.oberon) - 1)
        identity_prompt = INFO_YOUR_ROLE.format(self.name, self.role_name, verbal_side[self.side]) # and do not pretend to be other roles throughout the game."
        self.identity_prompt = identity_prompt

        # Reveal Prompt
        reveal_info = ''
        minion_list = []
        servant_list = []
        assassin = ''
        merlin = ''
        for idx, player_info in enumerate(player_list):
            if player_info[1] == "Minion":
                minion_list.append(str(idx))
            elif player_info[1] == "Servant":
                servant_list.append(str(idx))
            elif player_info[1] == "Assassin":
                assassin = str(idx)
            elif player_info[1] == "Merlin":
                merlin = str(idx)
        if self.role_name == "Merlin":
            if len(minion_list) == 1:
                reveal_info = REVEAL_PROMPTS['Merlin'][0].format(', '.join(minion_list), ', '.join(servant_list))
            elif len(minion_list) > 1:
                reveal_info = REVEAL_PROMPTS['Merlin'][1].format(', '.join(minion_list))
        if self.role_name == "Minion":
            if len(minion_list) == 1:
                reveal_info = REVEAL_PROMPTS['Minion'][0].format(assassin, ', '.join(servant_list + [merlin]))
            elif len(minion_list) > 1:
                reveal_info = REVEAL_PROMPTS['Minion'][1].format(', '.join(minion_list))
        if self.role_name == "Assassin":
            if len(minion_list) == 1:
                reveal_info = REVEAL_PROMPTS['Assassin'][0].format(', '.join(minion_list), ', '.join(servant_list + [merlin]))

        # Seperately pass the reveal info to the agent, so as to meet the requirement in filer_messages
        # TODO: is `system` allowed? 
        self.session.inject({
            "role": "user",
            "content": content_prompt,
            "mode": "system",
        })
        self.session.inject({
            # "role": "system",
            "role": "user",
            "content": identity_prompt + '\n' + reveal_info,
            "mode": "system",
        })
        self.system_info = content_prompt + '\n' + identity_prompt + '\n' + reveal_info
        self.dialogue_generator = DialogueGenerator(
            llm_model=GPT35Multi(),
            prompt_generator=self.prompt_generator,
            player=self.id,
            private_information=self.system_info
        )
        self.dialogue_discriminator = DialogueDiscriminator(
            llm_model=GPT35Multi(),
            prompt_generator=self.prompt_generator,
            known_sides=self.side,
            player=self.id,
            player_role=self.role,
            private_information=self.system_info
        )

    async def team_discussion(self, team_size, team_leader_id, mission_id, env=None, **kwargs):
        """
        Team discussion
        """
        history = kwargs.pop("history", "")
        self.dialogue_discriminator.update_beliefs(history=history)
        action = self.get_action_intent(env)
        state = self.convert_to_avalon_state(env)
        dialogue = self.dialogue_generator.generate_dialogue(
            intended_action=action,
            history=history,
            quest_leader=state.quest_leader,
            phase=state.phase,
            turn=state.turn,
            round=state.round,
            quest_team=state.quest_team,
            team_votes=state.team_votes,
            quest_results=state.quest_results,
        )
        return dialogue

    async def propose_team(self, team_size, mission_id, env=None):
        """
        Propose Team
        """
        return self.get_action_intent(env)
    
    async def vote_on_team(self, team, mission_id, env=None):
        """
        Vote on team
        """
        return self.get_action_intent(env)

    def convert_to_avalon_state(self, env=None) -> AvalonState:
        '''
        Converts the current Avalonbench game state to an AvalonState object
        '''
        return AvalonState.init_from_env(env)
    
    def get_action_intent(self, env=None):
        '''
        Returns the action intent of the agent
        '''
        state = self.convert_to_avalon_state(env)
        belief_p_is_good = self.dialogue_discriminator.get_p_is_good()
        belief_p_is_merlin = self.dialogue_discriminator.get_p_is_merlin()
        action = self.action_planner.get_intent(
            belief_p_is_good=belief_p_is_good,
            belief_p_is_merlin=belief_p_is_merlin,
            quest_leader=state.quest_leader,
            phase=state.phase,
            turn=state.turn,
            round=state.round,
            done=state.done,
            quest_team=state.quest_team,
            team_votes=state.team_votes,
            quest_votes=state.quest_votes,
            quest_results=state.quest_results,
            good_victory=state.good_victory
        )
        return action
    
    