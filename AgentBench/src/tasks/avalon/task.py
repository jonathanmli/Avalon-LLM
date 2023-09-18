import sys
import re
import time
from src.task import Task, Dataset, DataPiece
from .engine import AvalonGameEnvironment, AvalonConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import random
from copy import deepcopy
from typing import Dict, Callable, Type, Tuple, List, Any, Union, Iterable, Generic, TypeVar

from ...agent import Agent, Session
from .baseline_agents import NaiveAssassin, NaiveMerlin, NaiveMinion, NaiveServant

from .api import *
from .prompts import *
from .utils import *

T_INPUT = TypeVar('T_INPUT')
T_OUTPUT = TypeVar('T_OUTPUT')
T_TARGET = TypeVar('T_TARGET')



class player:

    def __init__(self, name, num_players, session, id, config:AvalonConfig, side=None, seed=None, **kwargs):
        self.name = name
        self.id = id
        self.num_players = num_players
        self.role = None
        self.team = None
        self.side = side # 1 for good, 0 for evil
        self.session = session

        self.merlin = kwargs.pop("merlin")
        self.percival = kwargs.pop("percival")
        self.morgana = kwargs.pop("morgana")
        self.mordred = kwargs.pop("mordred")
        self.oberon = kwargs.pop("oberon")

        self.num_good = kwargs.pop("num_good")
        self.num_evil = kwargs.pop("num_evil")
        self.player_list = kwargs.pop("player_list")

        self.seed = seed
        print("The seed is: ", self.seed)

        self.config = config
        self.strategy = None


    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
    
    def execute_tool(self, message, team_size=None):
        if self.session.name == "Random-Agent":
            return eval(message)

        lines = message.split("\n")
        find_action = False
        for line in lines:
            execution_message = "Function is not executed!"
            # if re.match(r"Action.*?:", line):
            # find_action = True
            function_names = re.findall(r'\w+\((?!\s*$).*\)', line)
            function_executed = False
            while len(function_names) != 1:
                intermediate_output = self.session.action({
                    "role": "user",
                    "content": "You'are using the tools in a wrong way. Please try again.",
                    "mode": "rectify",
                    "seed": self.seed,
                    "role_name": self.role_name
                })
                function_names = re.findall(r'\w+\((?!\s*$).*\)', intermediate_output)
            function_name = function_names[-1]
            # for function_name in function_names:
            while not function_executed:
                try:
                    print("test function name: ", function_name)
                    result = eval(function_name)
                    # Ensure size of teh team chosen is correct
                    if team_size is not None:
                        while len(result) != team_size:
                            intermediate_output = self.session.action({
                                "role": "user",
                                "content": f"You'are choosing a team with the wrong size. Please choose the team again using the tool. The proper size of team should be {team_size}",
                                "mode": "rectify",
                                "seed": self.seed,
                                "role_name": self.role_name
                            })
                            function_names = re.findall(r'\w+\((?!\s*$).*\)', intermediate_output)
                            function_name = function_names[-1]

                            result = eval(function_name)
                    else:
                        assert int(result) in [0, 1]

                    function_executed = True
                    return result
                except:
                    function_names = []
                    while len(function_names) != 1:
                        intermediate_output = self.session.action({
                            "role": "user",
                            "content": "You'are using the tools in a wrong way. Please try again.",
                            "mode": "rectify",
                            "seed": self.seed,
                            "role_name": self.role_name
                        })
                        function_names = re.findall(r'\w+\((?!\s*$).*\)', intermediate_output)
                    function_name = function_names[-1]

    def parse_result(self, message, team_size):
        print(message)
        return eval(message)
    
    def propose_team(self, team_size, discussion_history, mission_id, mode):
        if mode == "discussion":
            content_prompt = f"You are the leader in this round. Please make some statements with one sentence."
        elif mode == "action":
            # if len(discussion_history) > 0:
            #     # content_prompt = ' '.join(discussion_history) + ' ' + f"Action Phase. Please choose {team_size} players from player ids 0 to {self.num_players-1}"
            #     content_prompt = ' '.join(discussion_history) + ' ' + f"Please choose {team_size} players from player ids 0 to {self.num_players-1}"
            # else:
            content_prompt = f"Please choose {team_size} players from player ids 0 to {self.num_players-1} by using `choose` function."

        if self.strategy is not None:
            naive_result = self.strategy.propose_team(mission_id=mission_id)

        proposed_team = self.session.action({
            "role": "user",
            "content": content_prompt,
            "team_size": team_size,
            "mode": "choose_quest_team_" + mode,
            "seed": self.seed,
            "role_name": self.role_name,
            "naive_result": naive_result
        })
        print(proposed_team)

        # return self.execute_tool(proposed_team, team_size=team_size), get_statement(proposed_team)
        # return proposed_team, get_statement(proposed_team)
        if mode == "action":
            return eval(proposed_team), None
        elif mode == "discussion":
            return None, proposed_team
    
    def vote_on_team(self, team, statement_history, mission_id, mode="statement"):
        if mode == "statement":
            content_prompt = ' '.join(statement_history) + ' ' + f"Discussion Phase. Please vote on the team {team}."
        elif mode == "action":
            # content_prompt = f"Action Phase. Please vote on the team {team}."
            content_prompt = f"Please vote on the team {team} by using `vote` function."
        else:
            raise RuntimeError(
                f"Unexpected Mode {mode}."
            )
        
        if self.strategy is not None:
            naive_result = self.strategy.vote_on_team(mission_id=mission_id, team=team)

        print("Content Prompt: ", content_prompt)
        vote_result = self.session.action({
            "role": "user",
            "content": content_prompt,
            "side": int(self.side),
            "mode": "vote_on_team",
            "seed": self.seed,
            "role_name": self.role_name,
            "naive_result": naive_result
        })
        if mode == "statement":
            statement_history.append(f"Statements from {self.name}: {get_statement(vote_result)}")
        # print(statement_history)
        # return random.choice([0, 1])
        # return self.execute_tool(vote_result)
        return eval(vote_result)
    
    def vote_on_mission(self, statement_history, mission_id, team, mode="statement"):
        if mode == "statement":
            content_prompt = ' '.join(statement_history) + ' ' + f"Please vote on the quest."
        elif mode == "action":
            content_prompt = f"Please vote on the quest by using `vote` function."
        else:
            raise RuntimeError(
                f"Unexpected Mode {mode}."
            )
        
        if self.strategy is not None:
            naive_result = self.strategy.vote_on_mission(mission_id=mission_id, team=team)

        vote_result = self.session.action({
            "role": "user",
            "content": content_prompt,
            "side": int(self.side),
            "mode": "vote_on_mission",
            "seed": self.seed,
            "role_name": self.role_name,
            "naive_result": naive_result
        })
        # if mode == "statement":
        #     statement_history.append(f"Statements from {self.name}: {get_statement(vote_result)}")
        if mode == "action":
            statement_history.append(f"{self.name} votes {vote_result} on the mission")
        # return self.side
        # return self.execute_tool(vote_result)
        return eval(vote_result)
    
    def assign_side(self, side):
        sides = ['Evil', 'Good']
        self.side = side
        # self.session.inject({
        #     "role": "user",
        #     "content": f"You are on the {sides[side]} side."
        # })
        # self.session.inject({
        #     "role": "agent",
        #     "content": "I understand."
        # })

    def assign_role(self, role, role_name, sides):
        self.role = role
        self.role_name = role_name

        if role_name == "Merlin":
            self.strategy = NaiveMerlin(id=self.id, name=self.name, sides=sides, config=self.config)
        elif role_name == "Minion":
            self.strategy = NaiveMinion(id=self.id, name=self.name, sides=sides, config=self.config)
        elif role_name == "Assassin":
            self.strategy = NaiveAssassin(id=self.id, name=self.name, sides=sides, config=self.config)
        elif role_name == "Servant":
            self.strategy = NaiveServant(id=self.id, name=self.name, sides=sides, config=self.config)

        """
        Instruction Prompt
        """
        if role_name == "Assassin":
            INSTRUCTIONS = INSTRUCTIONS_ASSASSIN
        else:
            INSTRUCTIONS = INSTRUCTIONS_NORMAL
        content_prompt = INSTRUCTIONS + "\n" + f"There are {self.num_players} players, including Player 0, Player 1, Player 2, Player 3, and Player 4. {self.num_good} players are good, including {int(self.merlin)} Merlin, and {self.num_good - int(self.merlin) - int(self.percival)} Loyal Servant(s) of Arthur's. {self.num_evil} players are evil, 1 Assassin, and {self.num_evil - int(self.morgana) - int(self.mordred) - int(self.oberon) - 1} Minion(s) of Mordred."
        initialization_prompt = f"You are {self.name}, and you are {role_name}."
        self.session.inject({
            "role": "system",
            "content": content_prompt,
            "mode": "system",
        })
        self.session.inject({
            "role": "user",
            "content": initialization_prompt
        })
        self.session.inject({
            "role": "agent",
            "content": "I understand."
        })
        # self.session.inject({
        #     "role": "user",
        #     "content": INSTRUCTIONS
        # })
        # self.session.inject({
        #     "role": "agent",
        #     "content": "I understand."
        # })
        # self.session.inject({
        #     "role": "user",
        #     "content": f"There are {self.num_players} players. {self.num_good} players are good, including {int(self.merlin)} Merlin, {int(self.percival)} Percival, and {self.num_good - int(self.merlin) - int(self.percival)} Loyal Servant(s) of Arthur's. {self.num_evil} players are evil, including {int(self.morgana)} Morgana, {int(self.mordred)} Mordred, {int(self.oberon)} Oberon, 1 Assassin, and {self.num_evil - int(self.morgana) - int(self.mordred) - int(self.oberon) - 1} Minion(s) of Mordred."
        # })
        # self.session.inject({
        #     "role": "agent",
        #     "content": "I understand, please start."
        # })
        """
        One-shot Prompts
        """
        # if role_name == "Assassin":
        #     for i, prompt in enumerate(ONE_SHOT_ASSASSIN_NO_THOUGHT):
        #         if i % 2 == 0:
        #             self.session.inject({
        #                 "role": "user",
        #                 "content": prompt
        #             })
        #         else:
        #             self.session.inject({
        #                 "role": "agent",
        #                 "content": prompt
        #             })
        #     for i, prompt in enumerate(ONE_SHOT_ASSASSIN_DISCUSSION):
        #         if i % 2 == 0:
        #             self.session.inject({
        #                 "role": "user",
        #                 "content": prompt
        #             })
        #         else:
        #             self.session.inject({
        #                 "role": "agent",
        #                 "content": prompt
        #             })
        # elif role_name in ["Merlin", "Loyal Servant of Arthur's"]:
        #     for i, prompt in enumerate(ONE_SHOT_GOOD):
        #         if i % 2 == 0:
        #             self.session.inject({
        #                 "role": "user",
        #                 "content": prompt
        #             })
        #         else:
        #             self.session.inject({
        #                 "role": "agent",
        #                 "content": prompt
        #             })
        #     for i, prompt in enumerate(ONE_SHOT_GOOD_DISCUSSION):
        #         if i % 2 == 0:
        #             self.session.inject({
        #                 "role": "user",
        #                 "content": prompt
        #             })
        #         else:
        #             self.session.inject({
        #                 "role": "agent",
        #                 "content": prompt
        #             })
        # elif role_name in ["Minion of Modred"]:
        #     for i, prompt in enumerate(ONE_SHOT_EVIL):
        #         if i % 2 == 0:
        #             self.session.inject({
        #                 "role": "user",
        #                 "content": prompt
        #             })
        #         else:
        #             self.session.inject({
        #                 "role": "agent",
        #                 "content": prompt
        #             })
        #     for i, prompt in enumerate(ONE_SHOT_EVIL_DISCUSSION):
        #         if i % 2 == 0:
        #             self.session.inject({
        #                 "role": "user",
        #                 "content": prompt
        #             })
        #         else:
        #             self.session.inject({
        #                 "role": "agent",
        #                 "content": prompt
        #             })

        """
        Tutorial on strategies
        """
        # strategy_prompt = TUTORIAL_STRATEGIES_PROMPTS_ZERO_SHOT[role_name]
        # for prompt in strategy_prompt:
        #     self.session.inject({
        #         "role": "user",
        #         "content": prompt
        #     })
        #     self.session.inject({
        #         "role": "agent",
        #         "content": prompt
        #     })      

        """
        Reveal Phase
        """
        reveal_info = None
        if role_name == "Merlin":
            minion_list = []
            for idx, player_info in enumerate(self.player_list):
                if player_info[1] == "Minion of Modred":
                    minion_list.append(str(idx))
            if len(minion_list) == 1:
                reveal_info = REVEAL_PROMPTS['Merlin'][0].format(*minion_list)
            elif len(minion_list) > 1:
                reveal_info = REVEAL_PROMPTS['Merlin'][1].format(*minion_list)
        if role_name == "Minion":
            minion_list = []
            for idx, player_info in enumerate(self.player_list):
                if player_info[1] == "Minion of Modred":
                    minion_list.append(str(idx))
            if len(minion_list) == 1:
                reveal_info = REVEAL_PROMPTS['Minion of Modred'][0].format(*minion_list)
            elif len(minion_list) > 1:
                reveal_info = REVEAL_PROMPTS['Minion of Modred'][1].format(*minion_list)
        if reveal_info is not None:
            self.session.inject({
                "role": "user",
                "content": reveal_info
            })
            self.session.inject({
                "role": "agent",
                "content": "Okay, I understand"
            })

        """
        Thought: what is my strategy for this game?
        """
        # self.session.action({
        #     "role": "user",
        #     "content": f"Based on the tutorial, what is my strategy for this game?",
        #     "mode": "strategy"
        # })

    def assassinate(self, player):
        if player >= self.num_players:
            raise ValueError(f"Player {player} does not exist.")
        if self.role != 7:
            raise ValueError("Only assassin can assassinate.")
        
        # assassinate_result = self.session.action({
        #     "role": "user",
        #     "content": f"Action Phase. Assassination phase. Your job is to assassinate Merlin. \
        #         Choose a player (id) to assassinate. Choose the player id from 0 to {self.num_players-1}",
        #     "mode": "assassination"
        # })
        assassinate_result = self.session.action({
            "role": "user",
            "content": f"Assassination phase. Your job is to assassinate Merlin. \
                Choose a player (id) to assassinate. Choose the player id from 0 to {self.num_players-1}",
            "mode": "assassination",
            "seed": self.seed,
            "role_name": self.role_name,
        })
        # return self.execute_tool(assassinate_result)
        # return self.parse_result(assassinate_result)
        return eval(assassinate_result)


class Avalon(Task):
    def __init__(self, **configs):
        self.num_players = configs.pop('num_players')
        self.seed = configs.pop('seed')
        self.avalon_config = AvalonConfig(self.num_players, self.seed)
        self.env = AvalonGameEnvironment(self.avalon_config)
        super().__init__(**configs)

        self.llm_sides = []

    def get_current_agents():
        pass

    def get_data(self):
        info = Dataset()
        '''
        Plan A: generate data on the fly
        '''
        # data = self.env.__dict__
        data = []
        for i in range(0, 50):
            data = [i for _ in range(5)]
        # print(data)
            info.append(DataPiece(data, None))
        # info.append(DataPiece(data, None))
        print(info)
        return info
    
    @property
    # TODO: find some proper metrics for avalon
    def metrics(self) -> Dict[str, Callable[[List[T_OUTPUT], List[T_TARGET]], Any]]:
        print({"Success Rate": lambda x, y: sum(np.array(x) == np.array(self.llm_sides)) / len(x)})
        return {"Success Rate": lambda x, y: sum(np.array(x) == np.array(self.llm_sides)) / len(x)}  # Hack the metric
    
    def predict_all(self, agents: List[Agent], inputs: List[T_INPUT], already_runs: List[Any]=None) -> List[T_OUTPUT]:
        print(f"Start Predicting All ...")
        assert already_runs is None or len(already_runs) == len(inputs)

        thread_count = self.workers
        if self.worker_limit:
            thread_count = min(self.workers, self.worker_limit)

        executor = ThreadPoolExecutor(max_workers=thread_count)

        threads = []
        results = [None] * len(inputs)

        def call_wrap(data_item, index):
            try:
                sessions = []
                for agent in agents:
                    session = agent.create_session()
                    sessions.append(session)
                result = self.predict_single(sessions, data_item)
                self.save_single(index, data_item, result, session)
            except Exception as e:
                import traceback
                traceback.print_exc()
                pass
            results[index] = result

        for idx, item in enumerate(inputs):
            if already_runs is not None and already_runs[idx] is not None:
                results[idx] = already_runs[idx]
                continue
            future = executor.submit(call_wrap, item, idx)
            threads.append(future)
            # time.sleep(30)
        
        with tqdm(total=len(inputs)) as pbar:
            for thread in as_completed(threads):
                pbar.update(1)

        return results

    def predict_single(self, sessions, data_item):
        # ask for number of players
        # num_players = int(input("Enter number of players: "))
        # env = AvalonGameEnvironment(num_players)
        # print(sys.modules[__name__])
        num_players = self.num_players
        env = self.env
        env.reset()
        while env.get_roles()[1][1] != "Assassin":
            env.reset()

        # print("Data item: ", data_item)

        seeds = data_item

        player_list = []

        if num_players != len(sessions):
            raise ValueError(
                f"Number of players {num_players} doesn't match number of sessions {len(sessions)}"
            )


        for i in range(num_players):
            random.seed(self.seed, version=1)
            player_list.append(player(f"Player {i}",
                                    num_players,
                                    sessions[i],
                                    # random.choice([0, 1]),
                                    id = i,
                                    config = self.avalon_config,
                                    merlin = env.merlin,
                                    percival = env.percival,
                                    morgana = env.morgana,
                                    mordred = env.mordred,
                                    oberon = env.oberon,
                                    num_good = env.num_good,
                                    num_evil = env.num_evil,
                                    player_list = env.get_roles(),
                                    seed=seeds[i]
                                    ))

        print(env.get_roles())

        

        for i, (role_i, role_name, side) in enumerate(env.get_roles()):
            player_list[i].assign_role(role_i, role_name, env.get_partial_sides(i))
            player_list[i].assign_side(side)
            print(f"{player_list[i]} is {role_name}")

        print("Player role: ", player_list[1].role)
        if player_list[1].role in [6, 7]:
            self.llm_sides.append(-1)
        else:
            self.llm_sides.append(1)

        print("LLM Sides: ", self.llm_sides)

        while not env.done:
            # print phase from env
            phase = env.get_phase()[0]
            print(env.get_phase()[1])
            
            # if phase is team selection phase, ask for team
            if phase == 0:
                discussion_history = []
                leader = env.get_quest_leader()
                """
                Leader speaks
                """
                team, statement = player_list[leader].propose_team(env.get_team_size(), discussion_history, env.turn, mode="discussion")
                print(f"Please choose {env.get_team_size()} players in this round.")


                """
                Discussion (sequential, once, in order for now)
                """
                for idx, session in enumerate(sessions):
                    # if idx == leader:
                    #     continue
                    discussion = session.action({
                        "role": "user",
                        # "content": 'test',
                        "content": f"Statement from {player_list[leader]}: \n\"{statement}\"\nAnd words from other players:\n{' '.join(discussion_history)}\n This is discussion phase, and you don't need to take an action. Please discuss about words from the leader and other players with just one sentence.",
                        "mode": "discuss_on_team"
                    })
                    discussion_history.append(f"Player {idx} : " + discussion)
                for idx, session in enumerate(sessions):
                    session.inject({
                        "role": "user",
                        "content": f"Statement from {player_list[leader]}: \n\"{statement}\"\nAnd words from other players:\n{' '.join(discussion_history)}"
                    })
                    session.inject({
                        "role": "agent",
                        "content": "I understand."
                    })
                # votes_first_round = [player_list[i].vote_on_team(env.get_current_quest_team(), discussion_history, mode="statement"
                #                                      ) for i in range(num_players)]
                """
                Choose a team
                """
                team, statement = player_list[leader].propose_team(env.get_team_size(), discussion_history, env.turn, mode="action")
                print(team)
                env.choose_quest_team(team, leader)
                print(f"{player_list[leader]} proposed team {team}")

            # if phase is team voting phase, ask for votes
            elif phase == 1:
                discussion_history = []
                # votes_first_round = [player_list[i].vote_on_team(env.get_current_quest_team(), discussion_history, mode="statement"
                #                                      ) for i in range(num_players)]
                votes = [player_list[i].vote_on_team(env.get_current_quest_team(), discussion_history, env.turn, mode="action"
                                                    ) for i in range(num_players)]
                print(votes)
                outcome = env.vote_on_team(votes)
                print(f"Team votes: {votes}, team outcome: {outcome[2]}")
                # for session in sessions:
                #     session.action({
                #         "role": "user",
                #         "content": f"Mission outcome: {outcome[2]}",
                #         "mode": "system"
                #     })


            # if phase is quest voting phase, ask for votes
            elif phase == 2:
                discussion_history = []
                # votes_first_round = [player_list[i].vote_on_mission(discussion_history, mode="statement"
                #                                                     ) for i in env.get_current_quest_team()]
                votes = [player_list[i].vote_on_mission(discussion_history, env.turn, team, mode="action"
                                                        ) for i in env.get_current_quest_team()]
                outcome = env.vote_on_quest(votes)
                print(f"Quest votes: {votes}, mission outcome: {outcome[2]}")
                outcome_verbal = ["failed", "succeeded"]
                for idx, session in enumerate(sessions):
                    session.action({
                        "role": "user",
                        "content": f"This mission has {outcome_verbal[int(outcome[2])]} based on the quest results.",
                        "mode": "system",
                        "seed": self.seed,
                        "role_name": player_list[idx].role_name
                    })
                """
                Thought on quest and team result
                """
                # for session in sessions:
                #     session.action({
                #         "role": "user",
                #         "content": f"{player_list[leader]} proposed team {team}. What do you think?",
                #         "mode": "thought_on_quest_and_team_result"
                #     })
                    # session.inject({
                    #     "role": "agent",
                    #     "content": "I understand."
                    # })
                print(f"Testing Observe Mission at turn {env.turn-1}...")
                for idx, good in enumerate(env.is_good):
                    if good and env.turn < 5:
                        player_list[idx].strategy.observe_mission(team, env.turn-1, sum(np.array(votes) == 0))
            
            # if phase is assassination phase, ask for assassination
            elif phase == 3:
                # discussion_history = []
                # for session in sessions:
                #     discussion = session.action({
                #         "role": "user",
                #         "content": ' '.join(discussion_history) + ' ' + "Discussion Phase. Please discuss on the assassination target.",
                #         "mode": "vote"
                #     })
                #     discussion_history.append(f"{session.name}: " + discussion)


                assassin = env.get_assassin()
                target = int(player_list[assassin].assassinate(assassin))
                # target = int(input(f"Enter assassination target: "))
                print(f"Assassination target: {target}")
                env.choose_assassination_target(assassin, target)

        # print whether good or evil won
        if env.good_victory:
            print("Good wins!")
            return 1
        else:
            print("Evil wins!")
            return -1