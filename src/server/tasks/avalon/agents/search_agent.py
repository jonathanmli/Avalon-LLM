from .llm_with_discussion import LLMAgentWithDiscussion # this part is not correct yet
from ..engine import AvalonBasicConfig
from ..wrapper import AvalonSessionWrapper, Session
from search_src.Avalon.baseline_models_Avalon import AvalonState, AvalonLLMFunctionalValueHeuristic
from search_src.dialogue_improve.action_planner import AvalonActionPlannerAgent, BaselineAvalonActionPlannerAgent
from search_src.dialogue_improve.dialogue_generator import DialogueGenerator
from search_src.dialogue_improve.dialogue_discrimination import DialogueDiscriminator
from search_src.dialogue_improve.prompt_generator import PromptGenerator
from search_src.searchlightimprove.llm_utils.llm_api_models import GPT35Multi
from ..prompts import *
from ..engine import *

from good_examples.Avalon.value_heuristics.list import functions as value_heuristics
from good_examples.Avalon.dialogue_guide.list import guides as dialogue_guides

from ..utils import slice_out_new_dialogue
from ..utils import verbalize_team_result
from src.utils import ColorMessage

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
        self.func_str = func_str
        # self.prompt_generator = PromptGenerator(config=self.config)


    async def initialize_game_info(self, player_list, env=None, **kwargs) -> None:
        print(player_list)
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
        # self.dialogue_generator = DialogueGenerator(
        #     llm_model=GPT35Multi(),
        #     prompt_generator=self.prompt_generator,
        #     player=self.id,
        #     private_information=self.system_info
        # )
        # self.dialogue_discriminator = DialogueDiscriminator(
        #     llm_model=GPT35Multi(),
        #     prompt_generator=self.prompt_generator,
        #     known_sides=[int(player_info[2]) for player_info in player_list],
        #     player=self.id,
        #     player_role=self.role,
        #     players=[player_info[0] for player_info in player_list],
        #     private_information=self.system_info
        # )
        value_heuristic = AvalonLLMFunctionalValueHeuristic(func=self.func_str)

        known_sides = AvalonState.get_known_sides(self.id, [role[0] for role in player_list])
        dialogue_guide = dialogue_guides[self.id % len(dialogue_guides)] # I guess the id should just be an arbitrary number?
        role_to_dialogue_guide = {role: dialogue_guide for role in env.roles}
        # self.action_planner = AvalonActionPlannerAgent(
        self.action_planner = BaselineAvalonActionPlannerAgent(
            config=self.config,
            llm_model=GPT35Multi(),
            player=self.id,
            value_heuristic=value_heuristic,
            role_to_dialogue_guide=role_to_dialogue_guide
        )
        # self.action_planner = AvalonActionPlannerAgent(
        #     config=None,
        #     llm_model=None,
        #     player=None,
        #     value_heuristic=None,
        #     dialogue_guide=None
        # )

    async def observe_team_result(self, mission_id, team: frozenset, votes: List[int], outcome: bool, **kwargs) -> None:
        # self.session.inject()
        self.session.inject({
            "role": "user",
            "content": verbalize_team_result(team, votes, outcome),
        })

    async def team_discussion(self, team_size, team_leader_id, mission_id, env=None, dialogue_history=None, **kwargs):
        """
        Team discussion
        """
        # history = kwargs.pop("history", "")
        # self.dialogue_discriminator.update_beliefs(history=history)
        # action = self.get_action_intent(env)
        # state = self.convert_to_avalon_state(env)
        # dialogue = self.dialogue_generator.generate_dialogue(
        #     intended_action=action,
        #     history=history,
        #     quest_leader=state.quest_leader,
        #     phase=state.phase,
        #     turn=state.turn,
        #     round=state.round,
        #     quest_team=state.quest_team,
        #     team_votes=state.team_votes,
        #     quest_results=state.quest_results,
        # )
        state = self.convert_to_avalon_state(env)
        actor, legal_actions = self.action_planner.actor_action_enumerator.enumerate(state)
        known_sides, self_role = state.get_private_information(actor)
        state.self_role = self_role
        state.self_player = actor
        state.known_sides = known_sides
        # information_set = self.action_planner.information_function.get_information_set(state, actor)
        new_dialogue = slice_out_new_dialogue(dialogue_history.get_list(), player=self.id)
        self.action_planner.observe_dialogue(state=state, new_dialogue=new_dialogue)
        utterance = self.action_planner.produce_utterance(state=state)
        return utterance

    async def propose_team(self, team_size, mission_id, env=None):
        """
        Propose Team
        """
        action = list(self.get_action_intent(env))
        print(ColorMessage.blue("Action:") + " ", f"Propose team {action}")
        return action
    
    async def vote_on_team(self, team, mission_id, env=None):
        """
        Vote on team
        """
        verbal_team_act = {
            0: "Reject the team",
            1: "Approve the team",
        }
        action = self.get_action_intent(env)
        print(ColorMessage.blue("Action:") + " " + verbal_team_act[action])
        return action
    
    async def vote_on_mission(self, team, mission_id, env=None):
        """
        Vote on mission
        """
        verbal_team_act = {
            0: "Fail the mission",
            1: "Pass the mission",
        }
        action = self.get_action_intent(env)
        print(ColorMessage.blue("Action:") + " " + verbal_team_act[action])
        return action
    
    async def assassinate(self, env=None):
        """
        Assassinate
        """
        action = self.get_action_intent(env)
        print(ColorMessage.blue("Action:") + " ", f"Assassinate Player {action}")
        return action
    
    async def get_believed_sides(self, num_players: int, **kwargs) -> List[float]:
        """
        Get the believed sides of all players
        """
        belief_p_is_good = self.action_planner.dialogue_discriminator.get_p_is_good()
        return belief_p_is_good


    def convert_to_avalon_state(self, env=None) -> AvalonState:
        '''
        Converts the current Avalonbench game state to an AvalonState object
        '''
        return AvalonState.init_from_env(env)
    
    def get_action_intent(self, env=None):
        '''
        Returns the action intent of the agent
        '''
        # state = self.convert_to_avalon_state(env)
        # print("Checkpoint intent 2")
        # belief_p_is_good = self.dialogue_discriminator.get_p_is_good()
        # print("Checkpoint intent 3")
        # belief_p_is_merlin = self.dialogue_discriminator.get_p_is_merlin()
        # print("Checkpoint intent 4")
        # action = self.action_planner.get_intent(
        #     belief_p_is_good=belief_p_is_good,
        #     belief_p_is_merlin=belief_p_is_merlin,
        #     quest_leader=state.quest_leader,
        #     phase=state.phase,
        #     turn=state.turn,
        #     round=state.round,
        #     done=state.done,
        #     quest_team=state.quest_team,
        #     team_votes=state.team_votes,
        #     # quest_votes=state.quest_votes,
        #     quest_results=state.quest_results,
        #     good_victory=state.good_victory
        # )
        state = self.convert_to_avalon_state(env)
        if state.done == True:
            print("Game is done.")
        # actor = -1
        # while actor != self.id:
        actor, legal_actions = self.action_planner.actor_action_enumerator.enumerate(state)
        if actor != 0:
            print("Wrong Actor")
        # FIXME: should I use the get_information_set here? Am I passing the correct actor?
        # information_set = self.action_planner.information_function.get_information_set(state, actor)
        # information_set = self.action_planner.information_function.get_information_set(state, actor)
        action = self.action_planner.act(state, legal_actions)
        # action = self.action_planner.act(state, legal_actions)
        print()
        print(ColorMessage.cyan(f"##### Search Agent (Player {self.id}, Role: {self.role_name}) #####"))
        print()
        # mock_dialogue = [
        #     (0, "I think we should approve this team."),
        #     (1, "I think we should reject this team."),
        #     (2, "I think we should approve this team."),
        #     (3, "I think we should reject this team."),
        #     (4, "I think we should approve this team.")
        # ]
        # print("Test speaking...")
        # self.action_planner.observe_dialogue(state=information_set, new_dialogue=mock_dialogue)
        # utterance = self.action_planner.produce_utterance(state=information_set)
        # print("Utterance: ", utterance)
        return action
    
    