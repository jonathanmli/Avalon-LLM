import sys
import re
from src.task import Task, Dataset, DataPiece
from .engine import AvalonGameEnvironment
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random
from typing import Dict, Callable, Type, Tuple, List, Any, Union, Iterable, Generic, TypeVar

from ...agent import Agent, Session

from .api import *
from .prompts import *
from .utils import *

T_INPUT = TypeVar('T_INPUT')
T_OUTPUT = TypeVar('T_OUTPUT')
T_TARGET = TypeVar('T_TARGET')

INSTRUCTIONS = """You're an agent playing a game called The Resistance: Avalon.

The Resistance: Avalon is the game of hidden loyalty. Players are either Loyal Servants of Arther fighting for Goodness and honor or aligned with the Evil ways of Mordred.

Good wins the game by successfully completing three Quests. Evil wins if three Quests end in failure. Evil can also win by assassinating Merlin at game's end or if a Quest cannot be undertaken.

Players may make any claims during the game, at any point in the game. Discussion, deception, accusation, and logical deducation are all equally important in order for Good to prevail or Evil to rule the day.

During the game, you will be using the following tools to decide a specific action:

1. vote(approval: boolean)
You will be using this function to vote on a Team/Quest. When approval is `True`, it means you approve the Team/Quest. Otherwise, if approval is assigned `False`, it means you reject the Team/Quest.

2. choose(player_list: list[int])
When you become the leader who is required to build a team up, you need to pass a list of player ids to this function. They will be the mmebers in your team.

3. assassinate(player_id: int)
This function should accept the player id of the person whom you think is most likely to be Merlin.
"""



class player:

    def __init__(self, name, num_players, session, side=None, **kwargs):
        self.name = name
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

        session.inject({
            "role": "user",
            "content": INSTRUCTIONS
        })
        session.inject({
            "role": "agent",
            "content": "I understand."
        })
        session.inject({
            "role": "user",
            "content": f"There are {num_players} players. {self.num_good} players are good, including {int(self.merlin)} Merlin, {int(self.percival)} Percival, and {self.num_good - int(self.merlin) - int(self.percival)} Loyal Servant(s) of Arthur's. {self.num_evil} players are evil, including {int(self.morgana)} Morgana, {int(self.mordred)} Mordred, {int(self.oberon)} Oberon, 1 Assassin, and {self.num_evil - int(self.morgana) - int(self.mordred) - int(self.oberon) - 1} Minion(s) of Mordred."
        })
        session.inject({
            "role": "agent",
            "content": "I understand, please start."
        })

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
    
    def execute_tool(self, message):
        if self.session.name == "Random-Agent":
            return eval(message)

        lines = message.split("\n")
        find_action = False
        for line in lines:
            execution_message = "Function is not executed!"
            if re.match(r"Action.*?:", line):
                find_action = True
                function_names = re.findall(r'\w+\((?!\s*$).*\)', line)
                function_executed = False
                for function_name in function_names:
                    try:
                        print(function_name)
                        result = eval(function_name)
                        return result
                    except:
                        raise RuntimeError(
                            "Execution Error"
                        )
    
    def propose_team(self, team_size):
        proposed_team = self.session.action({
            "role": "user",
            "content": f"Action Phase. Please choose {team_size} players from player ids 0 to {self.num_players-1}",
            "team_size": team_size,
            "mode": "choose_quest_team"
        })
        return self.execute_tool(proposed_team), get_statement(proposed_team)
    
    def vote_on_team(self, team, statement_history, mode="statement"):
        if mode == "statement":
            content_prompt = ' '.join(statement_history) + ' ' + f"Discussion Phase. Please vote on the team {team}."
        elif mode == "action":
            content_prompt = f"Action Phase. Please vote on the team {team}."
        else:
            raise RuntimeError(
                f"Unexpected Mode {mode}."
            )
        vote_result = self.session.action({
            "role": "user",
            "content": content_prompt,
            "side": int(self.side),
            "mode": "vote"
        })
        if mode == "statement":
            statement_history.append(f"Statements from {self.name}: {get_statement(vote_result)}")
        # print(statement_history)
        # return random.choice([0, 1])
        return self.execute_tool(vote_result)
    
    def vote_on_mission(self, statement_history, mode="statement"):
        if mode == "statement":
            content_prompt = ' '.join(statement_history) + ' ' + f"Please vote on the quest."
        elif mode == "action":
            content_prompt = f"Please vote on the quest."
        else:
            raise RuntimeError(
                f"Unexpected Mode {mode}."
            )
        vote_result = self.session.action({
            "role": "user",
            "content": content_prompt,
            "side": int(self.side),
            "mode": "vote"
        })
        if mode == "statement":
            statement_history.append(f"Statements from {self.name}: {get_statement(vote_result)}")
        # return self.side
        return self.execute_tool(vote_result)
    
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

    def assign_role(self, role, role_name):
        self.role = role
        self.session.inject({
            "role": "user",
            "content": f"You are {self.name}, and you are {role_name}."
        })
        self.session.inject({
            "role": "agent",
            "content": "I understand."
        })
        """
        One-shot Prompts
        """
        if role_name == "Assassin":
            for i, prompt in enumerate(ONE_SHOT_ASSASSIN):
                if i % 2 == 0:
                    self.session.inject({
                        "role": "user",
                        "content": prompt
                    })
                else:
                    self.session.inject({
                        "role": "agent",
                        "content": prompt
                    })
            for i, prompt in enumerate(ONE_SHOT_ASSASSIN_DISCUSSION):
                if i % 2 == 0:
                    self.session.inject({
                        "role": "user",
                        "content": prompt
                    })
                else:
                    self.session.inject({
                        "role": "agent",
                        "content": prompt
                    })
        elif role_name in ["Merlin", "Loyal Servant of Arthur's"]:
            for i, prompt in enumerate(ONE_SHOT_GOOD):
                if i % 2 == 0:
                    self.session.inject({
                        "role": "user",
                        "content": prompt
                    })
                else:
                    self.session.inject({
                        "role": "agent",
                        "content": prompt
                    })
            for i, prompt in enumerate(ONE_SHOT_GOOD_DISCUSSION):
                if i % 2 == 0:
                    self.session.inject({
                        "role": "user",
                        "content": prompt
                    })
                else:
                    self.session.inject({
                        "role": "agent",
                        "content": prompt
                    })
        elif role_name in ["Minion of Modred"]:
            for i, prompt in enumerate(ONE_SHOT_EVIL):
                if i % 2 == 0:
                    self.session.inject({
                        "role": "user",
                        "content": prompt
                    })
                else:
                    self.session.inject({
                        "role": "agent",
                        "content": prompt
                    })
            for i, prompt in enumerate(ONE_SHOT_EVIL_DISCUSSION):
                if i % 2 == 0:
                    self.session.inject({
                        "role": "user",
                        "content": prompt
                    })
                else:
                    self.session.inject({
                        "role": "agent",
                        "content": prompt
                    })

        """
        Tutorial on strategies
        """
        strategy_prompt = TUTORIAL_STRATEGIES_PROMPTS_ZERO_SHOT[self.name]
        for prompt in strategy_prompt:
            self.session.inject({
                "role": "user",
                "content": prompt
            })
            self.session.inject({
                "role": "agent",
                "content": prompt
            })        

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
        if role_name == "Minion of Modred":
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
        self.session.action({
            "role": "user",
            "content": f"Based on the tutorial, what is my strategy for this game?"
        })

    def assassinate(self, player):
        if player >= self.num_players:
            raise ValueError(f"Player {player} does not exist.")
        if self.role != 7:
            raise ValueError("Only assassin can assassinate.")
        
        assassinate_result = self.session.action({
            "role": "user",
            "content": f"Action Phase. Assassination phase. Your job is to assassinate Merlin. \
                Choose a player (id) to assassinate. Choose the player id from 0 to {self.num_players-1}",
            "mode": "assassination"
        })
        return self.execute_tool(assassinate_result)


class Avalon(Task):
    def __init__(self, **configs):
        self.num_players = configs.pop('num_players')
        self.seed = configs.pop('seed')
        self.env = AvalonGameEnvironment(self.num_players, self.seed)
        super().__init__(**configs)

    def get_current_agents():
        pass

    def get_data(self):
        info = Dataset()
        '''
        Plan A: generate data on the fly
        '''
        data = self.env.__dict__
        info.append(DataPiece(data, None))
        print(info)
        return info
    
    @property
    # TODO: find some proper metrics for avalon
    def metrics(self) -> Dict[str, Callable[[List[T_OUTPUT], List[T_TARGET]], Any]]:
        return {"Success Rate": lambda x, y: len(x) + len(y)}  # Hack the metric
    
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
                                    random.choice([0, 1]),
                                    merlin = env.merlin,
                                    percival = env.percival,
                                    morgana = env.morgana,
                                    mordred = env.mordred,
                                    oberon = env.oberon,
                                    num_good = env.num_good,
                                    num_evil = env.num_evil,
                                    player_list = env.get_roles()
                                    ))

        print(env.get_roles())

        for i, (role_i, role_name, side) in enumerate(env.get_roles()):
            player_list[i].assign_role(role_i, role_name)
            player_list[i].assign_side(side)
            print(f"{player_list[i]} is {role_name}")

        while not env.done:
            # print phase from env
            phase = env.get_phase()[0]
            print(env.get_phase()[1])
            
            # if phase is team selection phase, ask for team
            if phase == 0:
                leader = env.get_quest_leader()
                team, statement = player_list[leader].propose_team(env.get_team_size())
                print(f"Please choose {env.get_team_size()} players in this round.")
                env.choose_quest_team(team, leader)
                print(f"{player_list[leader]} proposed team {team}")
                for session in sessions:
                    session.inject({
                        "role": "user",
                        "content": f"{player_list[leader]} proposed team {team}. Statement from Player {leader}: {statement}"
                    })
                    session.inject({
                        "role": "agent",
                        "content": "I understand."
                    })
            
            # if phase is team voting phase, ask for votes
            elif phase == 1:
                statement_history = []
                votes_first_round = [player_list[i].vote_on_team(env.get_current_quest_team(), statement_history, mode="statement"
                                                     ) for i in range(num_players)]
                votes = [player_list[i].vote_on_team(env.get_current_quest_team(), statement_history, mode="action"
                                                     ) for i in range(num_players)]
                outcome = env.vote_on_team(votes)
                print(f"Team votes: {votes}, team outcome: {outcome[2]}")

            # if phase is quest voting phase, ask for votes
            elif phase == 2:
                statement_history = []
                votes_first_round = [player_list[i].vote_on_mission(statement_history, mode="statement"
                                                                    ) for i in env.get_current_quest_team()]
                votes = [player_list[i].vote_on_mission(statement_history, mode="action"
                                                        ) for i in env.get_current_quest_team()]
                outcome = env.vote_on_quest(votes)
                print(f"Quest votes: {votes}, mission outcome: {outcome[2]}")
                for session in sessions:
                    session.inject({
                        "role": "user",
                        "content": f"{player_list[leader]} proposed team {team}"
                    })
                    session.inject({
                        "role": "agent",
                        "content": "I understand."
                    })

            # if phase is assassination phase, ask for assassination
            elif phase == 3:
                assassin = env.get_assassin()
                target = player_list[assassin].assassinate()
                # target = int(input(f"Enter assassination target: "))
                print(f"Assassination target: {target}")
                env.choose_assassination_target(assassin, target)

        # print whether good or evil won
        if env.good_victory:
            print("Good wins!")
        else:
            print("Evil wins!")