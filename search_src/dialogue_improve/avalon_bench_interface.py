from AvalonBench import LLMAgentWithDiscussion # this part is not correct yet
from Avalon.baseline_models_Avalon import AvalonState
from .action_planner import ActionPlanner
from .dialogue_generator import DialogueGenerator
from .dialogue_discrimination import DialogueDiscriminator
from .prompt_generator import PromptGenerator
from searchlightimprove.llm_utils.llm_api_models import GPT35Multi
from .data_loader import DataLoader

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
        private_information = ""
        self.action_planner = ActionPlanner(
            config=self.config,
            func_str=func_str,
            player=self.id,
            player_role=self.role,
            known_sides=self.side
        )
        self.prompt_generator = PromptGenerator()
        self.dialogue_generator = DialogueGenerator(
            llm_model=GPT35Multi(),
            prompt_generator=self.prompt_generator,
            player=self.id,
            private_information=private_information
        )
        self.dialogue_discriminator = DialogueDiscriminator(
            llm_model=GPT35Multi(),
            prompt_generator=self.prompt_generator,
            known_sides=self.side,
            player=self.id,
            player_role=self.role,
            private_information=private_information
        )

    async def team_discussion(self, team_size, team_leader_id, mission_id, env=None):
        """
        Team discussion
        """
        data_loader = DataLoader()
        data_loader.add_data_point(DISCUSSION_HISTORY, STATE_TUPLE, ACTION_INTENTS, PRIVATE_INFORMATIONS, ROLES, DIALOGUE, SPEAKER_ORDER)
        self.dialogue_discriminator.update_beliefs(history=)
        action = self.get_action_intent(env)
        dialogue = self.dialogue_generator.generate_dialogue(
            intended_action=action,
            history=,
        )

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
    
    