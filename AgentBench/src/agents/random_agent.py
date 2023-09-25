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
from copy import deepcopy
import openai
from ..tasks.avalon.utils import openai_wrapper

from ..tasks.avalon.arguments import args


class RandomAgent(Agent):
    """This agent is a random agent for avalon"""

    # def __init__(self,  seed=None, **kwargs) -> None:
    #     super().__init__(**kwargs)
        
    def __init__(self, api_args=None, num_players=None, **config):
        self.name = config.pop("name")
        self.num_players = num_players
        if not api_args:
            api_args = {}
        print("api_args={}".format(api_args))
        print("config={}".format(config))
        
        api_args = deepcopy(api_args)
        api_key = api_args.pop("key", None) or os.getenv('OPENAI_API_KEY')
        api_args["model"] = api_args.pop("model", None)
        api_args["api_key"] = api_key
        if not api_key:
            raise ValueError("OpenAI API key is required, please assign api_args.key or set OPENAI_API_KEY environment variable.")
        os.environ['OPENAI_API_KEY'] = api_key
        print("OpenAI API key={}".format(openai.api_key))
        api_base = api_args.pop("base", None) or os.getenv('OPENAI_API_BASE')
        os.environ['OPENAI_API_BASE'] = api_base
        print("openai.api_base={}".format(openai.api_base))
        if not api_args["model"]:
            raise ValueError("OpenAI model is required, please assign api_args.model.")
        self.api_args = api_args
        super().__init__(**config)

    def inference(self, history: List[dict]) -> str:
        '''
        There are 3 situations:
        1. choose quest team
        2. vote
        3. assassination
        '''
        # print(history)
        mode = history[-1]["mode"]
        seed = None if "seed" not in history[-1] else history[-1]["seed"]
        # TODO: should add the mode name here when using naive results
        if mode in ["choose_quest_team_action", "vote_on_team", "vote_on_mission", "assassination", "get_believed_sides"]:
            assert "naive_result" in history[-1]
            naive_result = history[-1]["naive_result"]
        else:
            naive_result = ''
        history = json.loads(json.dumps(history))
        for h in history:
            h.pop("mode", None)
            h.pop("team_size", None)
            h.pop("side", None)
            h.pop("seed", None)
            h.pop("role_name", None)
            h.pop("naive_result", None)
            if h['role'] == 'agent':
                h['role'] = 'assistant'

        # random.seed(seed, version=1)

        # if role_name == "Assassin":
        #     agent = NaiveAssassin
        # elif role_name == "Merlin":
        #     agent = NaiveMerlin
        # elif role_name == "Minion":
        #     agent = NaiveMinion
        # else:

        summary = []


            
        if mode == "choose_quest_team_action":
            print("Using Naive Strategy to Choose Quest Team...")
            # team_size = history[-1]["team_size"]
            # return str(random.sample(range(0, self.num_players), team_size))
            return frozenset(naive_result), summary, frozenset(naive_result)
        
        elif mode == "vote_on_team":
            # side = history[-1]["side"]
            # return str(random.choice([0, 1]))
            # return str(side)
            # print("Using Naive Strategy to Vote on Team...")
            return naive_result, summary, naive_result
        
        elif mode == "vote_on_mission":
            # print("Using Naive Strategy to Vote on Mission...")
            return naive_result, summary, naive_result
        
        elif mode == "assassination":
            # return random.randint(0, self.num_players-1), summary, random.randint(0, self.num_players-1)
            return naive_result, summary, naive_result
        
        elif mode == "strategy":
            return "None", summary, "None"
        
        elif mode == "discuss_on_team":
            if args.naive_summary == "full-history":
                """
                Summarize
                """
                summary_prompt = {
                    "role": "user",
                    "content": "Please summarize the history. Try to keep all the useful information, including your identification and your observations of the game."
                }
                summary_result = openai_wrapper(
                    messages=history[:-1] + [summary_prompt],
                    temperature=0.1,
                    **self.api_args
                )
                summary_result = summary_result["choices"][0]["message"]["content"]
                summary.append({
                    "role": "user",
                    "content": "Summary of previous information",
                    "mode": "summary"
                })
                summary.append({
                    "role": "agent",
                    # "content": summary_result,
                    "content": summary_result,
                    "mode": "summary"
                })

            elif args.naive_summary == "10-last":
                """
                Keep 10-last history
                """
                summary = history[-11:-1]
            """
            Discuss
            """
            resp = openai_wrapper(
                messages=[history[0]] + summary + [history[-1]],
                temperature=0.1,
                **self.api_args
            )
            resp = resp["choices"][0]["message"]["content"]
            result = resp
            return result, summary, result
        
        elif mode == "system":
            return "Okay", summary, "Okay"
        
        elif mode == "choose_quest_team_discussion":
            resp = openai_wrapper(
                messages=history,
                temperature=0.1,
                **self.api_args
            )
            resp = resp["choices"][0]["message"]["content"]
            result = resp

            return result, summary, result
        elif mode == "get_believed_sides":
            return naive_result, summary, naive_result

        else:
            raise NotImplementedError(
                f"There should not be other situations: {mode}."
            )