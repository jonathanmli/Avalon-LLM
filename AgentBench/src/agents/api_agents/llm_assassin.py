import openai
from src.agent import Agent
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
from typing import List, Callable
import dataclasses
from copy import deepcopy

ONE_SHOT_ASSASSIN_NO_THOUGHT = ["Tutorial of taking actions by thinking and using tools during action phase.",
                    "Okay, please start.",
                    "Please choose 3 players from player ids 0 to 4.",
                    "Action: choose([0, 2, 3]).",
                    "Please vote on the team [1, 3, 4].",
                    "Action: vote(True).",
                    "Please vote on the quest.",
                    "Action: vote(True).",
                    "Assassination phase. Your job is to assassinate Merlin. Choose a player (id) to assassinate. Choose the player id from 0 to 9.",
                    "Action: assassinate(3).",
                    "That's all about the tutorial.",                    
                    "OKay, I understand."
                    ]


class OpenAIChatCompletionAssassin(Agent):
    def __init__(self, api_args=None, **config):
        self.name = config.pop("name")
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
        """
        Summarize-then-action
        """
        print(history)
        mode = history[-1]["mode"]
        history = json.loads(json.dumps(history))
        for h in history:
            h.pop("mode", None)
            h.pop("team_size", None)
            h.pop("side", None)
            if h['role'] == 'agent':
                h['role'] = 'assistant'

        print("Mode: ", mode)
        if mode != "system":
            system_prompts = []
            for i, prompt in enumerate(ONE_SHOT_ASSASSIN_NO_THOUGHT):
                if i % 2 == 0:
                    system_prompts.append({
                        "role": "user",
                        "content": prompt
                    })
                else:
                    system_prompts.append({
                        "role": "assistant",
                        "content": prompt
                    })
        
            # Summarize
            summary_prompt = {
                "role": "user",
                "content": "Please summarize the history. Try to keep all the useful information, including your identification and your observations of the game."
            }
            summary_result = openai.ChatCompletion.create(
                messages=history[:-1] + [summary_prompt],
                temperature=0.7,
                **self.api_args
            )
            summary_result = summary_result["choices"][0]["message"]["content"]
            print("Summary: ", summary_result)

            # Action
            action_prompt = {
                "role": "user",
                "content": summary_result + "Please take action (choose or vote) using the tools based on the tutorial and the summary" + '\n' + history[-1]['content']
            }

            print(system_prompts)
            print(action_prompt)
            resp = openai.ChatCompletion.create(
                messages=system_prompts + [action_prompt],
                temperature=0,
                **self.api_args
            )
        else:
            resp = openai.ChatCompletion.create(
                messages=history,
                temperature=0,
                **self.api_args
            )

        # final_prompt = [{
        #     "role": "user",
        #     "content": summary_result + ' ' + history[-1]['content']
        # },
        # {
        #     "role": "assistant",
        #     "content": resp
        # },
        # {
        #     "role": "user",
        #     "content": "Therefore, your final choices is:"
        # }]

        # resp = openai.ChatCompletion.create(
        #     messages=final_prompt,
        #     **self.api_args
        # )
        print(resp)

        # return resp["choices"][0]["message"]["content"]

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