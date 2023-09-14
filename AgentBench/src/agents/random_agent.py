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
        random.seed(seed, version=1)
        if mode == "choose_quest_team":
            team_size = history[-1]["team_size"]
            return str(random.sample(range(0, self.num_players), team_size))
        elif mode == "vote":
            side = history[-1]["side"]
            # return str(random.choice([0, 1]))
            return str(side)
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