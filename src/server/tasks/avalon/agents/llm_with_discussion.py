from typing import List, Dict, Tuple
from .agent import Agent
from ..engine import AvalonBasicConfig
from ..wrapper import AvalonSessionWrapper, Session
from ..prompts import *
from copy import deepcopy
from ..utils import verbalize_team_result, verbalize_mission_result
from src.utils import ColorMessage
import logging

class LLMAgentWithDiscussion(Agent):
    r"""LLM agent with the ability to discuss with other agents."""

    def __init__(self, name: str, num_players: int, id: int, role: int, role_name: str, config:AvalonBasicConfig, session: AvalonSessionWrapper=None, side=None, seed=None, **kwargs):
        self.name = name
        self.id = id
        self.num_players = num_players
        self.role = role
        self.role_name = role_name
        self.side = side # 1 for good, 0 for evil
        self.session = session
        self.discussion = kwargs.pop('discussion', None)
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.seed = seed

        self.config = config

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
    
    def see_sides(self, sides):
        self.player_sides = sides
    
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

    async def summarize(self, **kwargs) -> None:
        # print("Summary")
        # self.session.inject()
        summary = await self.session.action({
            "role": "user",
            "content": "Please summarize the history. Try to keep all useful information, including your identity, other player's identities, and your observations in the game.",
            "mode": "summarize"
        })
        # print("Summary: ", summary)
        logging.info(f"Summary: {summary}", )
        past_history = deepcopy(self.session.get_history())
        self.session.overwrite_history([])
        self.session.inject({
            'role': "user",
            'content': self.system_info
        })
        self.session.inject({
            'role': "user",
            'content': summary
        })
        # print("History after summarization: ", self.session.get_history())
        return self.session.get_history()

    async def observe_mission(self, team, mission_id, num_fails, votes, outcome, **kwargs) -> None:
        pass

    async def observe_team_result(self, mission_id, team: frozenset, votes: List[int], outcome: bool, **kwargs) -> None:
        # self.session.inject()
        await self.session.action({
            "role": "user",
            "content": verbalize_team_result(team, votes, outcome),
        })
    
    async def get_believed_sides(self, num_players: int, **kwargs) -> List[float]:
        input = {
            "role": "user",
            "content": "To what extend do you believe each player to be Good, from Player 0 to Player 4? Please output probabilities within [0, 1] and round to two decimal places. If you are not sure, you can simply output 0.5.",
            "mode": "get_believed_sides",
        }
        # self.session.inject(input)
        believed_player_sides = await self.session.action(input)

        believed_player_sides = await self.session.parse_result(
            input   =   input,
            result  =   believed_player_sides
        )
        if isinstance(believed_player_sides, str):
            believed_player_sides = eval(believed_player_sides)
        print("Sides: ", believed_player_sides)
        return believed_player_sides

    # async def discussion_end(self):
    #     content_prompt = f"Discussion has ended. Here are the contents:\nStatement from Leader {leader}: \n\"{leader_statement}\"\nAnd words from other players:\n{' '.join(discussion_history)}"
    #     self.session.inject({
    #         "role": "user",
    #         "content": content_prompt,
    #     })

    async def team_discussion(self, team_size, team_leader_id, mission_id, **kwargs):
        """Team discussion phase.

        We also summarize the history before this phase at each round. If there's no discussion phase, we summarize the history before the vote phase.
        """
        # print("Discussion")
        # await self.summarize()

        fails_required = self.config.num_fails_for_quest[mission_id]
        content_prompt = CHOOSE_TEAM_LEADER + DISCUSSION_SUFFIX
        if self.id == team_leader_id:
            self.session.inject({
                "role": "user",
                "content": content_prompt,
            })
        else:
            self.session.inject({
                "role": "user",
                "content": DISCUSSION_SUFFIX
            })

        dialogue = await self.session.action(receiver="all")
        print(f"Output: {dialogue}")
        return dialogue


    async def quest_discussion(self, team_size, team, team_leader_id, discussion_history, mission_id, **kwargs):
        fails_required = self.config.num_fails_for_quest[mission_id]

    
    async def propose_team(self, team_size, mission_id, **kwargs):
        content_prompt = CHOOSE_TEAM_ACTION.format(team_size, self.num_players-1)

        thought = COTHOUGHT_PROMPT
        input = {
            "role": "user",
            "content": content_prompt + '\n' + thought,
            "team_size": team_size,
            "seed": self.seed,
            "role_name": self.role_name,
            "mode": "choose_quest_team_action",
        }
        # self.session.inject(input)
        proposed_team = await self.session.action(input)

        print()
        print(ColorMessage.cyan(f"##### LLM Agent (Player {self.id}, Role: {self.role_name}) #####"))
        print()
        print(ColorMessage.blue("Thought:") + " " + proposed_team)

        if isinstance(self.session.session, Session):
            proposed_team = await self.session.parse_result(input, proposed_team)
            proposed_team = eval(proposed_team)
        proposed_team = frozenset(proposed_team)
        print("Proposed Team: ", proposed_team)

        if isinstance(proposed_team, frozenset):
            return proposed_team
        else:
            raise ValueError(
                "Type of proposed_team must be frozenset, instead of {}.".format(type(proposed_team))
            )
        
    
    async def vote_on_team(self, team, mission_id, **kwargs):
        """Vote to approve or reject a team.

        If there's discussion phase, we will summarize the history before the vote phase.
        """
        if self.discussion:
            await self.summarize()

        content_prompt = VOTE_TEAM_ACTION.format(list(team))
        
        thought = COTHOUGHT_PROMPT
        input = {
            "role": "user",
            "content": content_prompt + "\n" + thought,
            "side": int(self.side),
            "mode": "vote_on_team",
            "seed": self.seed,
            "role_name": self.role_name,
        }
        # self.session.inject(input)
        vote_result = await self.session.action(input)

        print()
        print(ColorMessage.cyan(f"##### LLM Agent (Player {self.id}, Role: {self.role_name}) #####"))
        print()
        print(ColorMessage.blue("Thought:") + " " + vote_result)

        if isinstance(self.session.session, Session):
            vote_result = await self.session.parse_result(input, vote_result)
        vote_result = int(vote_result)

        if isinstance(vote_result, int):
            return vote_result
        else:
            raise ValueError(
                "Vote result should be either 0 or 1, instead of {}.".format(type(vote_result))
            )
    
    async def vote_on_mission(self, team, mission_id, **kwargs):
        content_prompt = VOTE_MISSION_ACTION.format(list(team))

        thought = COTHOUGHT_PROMPT
        input = {
            "role": "user",
            "content": content_prompt + "\n" + thought,
            "side": int(self.side),
            "mode": "vote_on_mission",
            "seed": self.seed,
            "role_name": self.role_name,
        }
        # self.session.inject(input)
        vote_result = await self.session.action(input)

        print()
        print(ColorMessage.cyan(f"##### LLM Agent (Player {self.id}, Role: {self.role_name}) #####"))
        print()
        print(ColorMessage.blue("Thought:") + " " + vote_result)

        if isinstance(self.session.session, Session):
            vote_result = await self.session.parse_result(input, vote_result)

        vote_result = int(vote_result)
        if isinstance(vote_result, int):
            return vote_result
        else:
            raise ValueError(
                "Vote result should be either 0 or 1, instead of {}.".format(type(vote_result))
            )
        

    async def assassinate(self, **kwargs):
        if self.role != 7:
            raise ValueError("Only the Assassin can assassinate.")
        
        thought = COTHOUGHT_PROMPT
        input = {
            "role": "user",
            "content": ASSASSINATION_PHASE.format(self.num_players-1) + "\n" + thought,
            "mode": "assassination",
            "seed": self.seed,
            "role_name": self.role_name,
        }
        # self.session.inject(input)
        assassinate_result = await self.session.action(input)
        # assassinate_result = int(assassinate_result)

        print()
        print(ColorMessage.cyan(f"##### LLM Agent (Player {self.id}, Role: {self.role_name}) #####"))
        print()
        print(ColorMessage.blue("Thought:") + " " + assassinate_result)

        if isinstance(self.session.session, Session):
            assassinate_result = await self.session.parse_result(input, assassinate_result)
            assassinate_result = int(assassinate_result)

        if isinstance(assassinate_result, int):
            return assassinate_result
        else:
            raise ValueError(
                "Assassination result should be an integer, instead of {}.".format(type(assassinate_result))
            )