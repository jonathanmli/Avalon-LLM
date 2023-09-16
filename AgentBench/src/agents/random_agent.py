from src.agent import Agent
from typing import List, Dict, Any, Optional, Union, Tuple, Callable, Type, TypeVar
import os
import json
import sys
import time
import re
import math
import random
import datetime
import argparse
import requests


class RandomAgent(Agent):
    """This agent is a random agent for avalon"""

    def __init__(self, num_players=None, seed=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_players = num_players
        self.name = kwargs.pop("name")

    def inference(self, history: List[dict]) -> str:
        '''
        There are 3 situations:
        1. choose quest team
        2. vote
        3. assassination
        '''
        # print(history)
        mode = history[-1]["mode"]
        seed = history[-1]["seed"]
        naive_result = None if "naive_result" not in history[-1] else history[-1]["naive_result"]


        random.seed(seed, version=1)
        # if role_name == "Assassin":
        #     agent = NaiveAssassin
        # elif role_name == "Merlin":
        #     agent = NaiveMerlin
        # elif role_name == "Minion":
        #     agent = NaiveMinion
        # else:
            
        if mode == "choose_quest_team":
            print("Using Naive Strategy to Choose Quest Team...")
            # team_size = history[-1]["team_size"]
            # return str(random.sample(range(0, self.num_players), team_size))
            return str(naive_result)
        elif mode == "vote_on_team":
            # side = history[-1]["side"]
            # return str(random.choice([0, 1]))
            # return str(side)
            print("Using Naive Strategy to Vote on Team...")
            return str(naive_result)
        elif mode == "vote_on_mission":
            print("Using Naive Strategy to Vote on Mission...")
            return str(naive_result)
        elif mode == "assassination":
            return str(random.randint(0, self.num_players-1))
        elif mode == "strategy":
            return "None"
        elif mode == "discuss_on_team":
            return "No idea."
        elif mode == "system":
            return "Okay"
        else:
            raise NotImplementedError(
                "There should not be other situations."
            )