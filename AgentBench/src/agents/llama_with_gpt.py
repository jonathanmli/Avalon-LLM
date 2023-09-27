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

from ..tasks.avalon.utils import openai_wrapper

import requests

def ollama_wrapper(history: List[dict]):

    # Define the base URL of your FastAPI application
    base_url = "http://172.31.76.7:8000"  # Replace with your actual server's address and port

    data = {"messages": history}

    response = requests.post(base_url + "/api", json=data)
    print("POST Response:", response.status_code, response.json())

    result = response.json()['result']['content']

    return result


class OpenAIChatCompletion(Agent):
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
        print(history)
        mode = history[-1]["mode"]
        naive_result = '' if "naive_result" not in history[-1] else history[-1]["naive_result"]
        history = json.loads(json.dumps(history))
        for h in history:
            h_mode = h.pop("mode", None)
            h.pop("team_size", None)
            h.pop("side", None)
            h.pop("seed", None)
            h.pop("role_name", None)
            h.pop("naive_result", None)
            if h['role'] == 'agent':
                h['role'] = 'assistant'
        # resp = openai.ChatCompletion.create(
        #     messages=history,
        #     **self.api_args
        # )

        # return resp["choices"][0]["message"]["content"]

        summary = []


            
        if mode == "choose_quest_team_action":

            return naive_result, summary, naive_result, self.player_id
        
        elif mode == "vote_on_team":

            result = ollama_wrapper(history=history)

            return naive_result, summary, naive_result, self.player_id
        
        elif mode == "vote_on_mission":

            return naive_result, summary, naive_result, self.player_id
        
        elif mode == "assassination":

            return naive_result, summary, naive_result, self.player_id
        
        elif mode == "strategy":
            return "None", summary, "None", self.player_id
        
        elif mode == "summarize":
            if args.naive_summary == "full-history":
                """
                Summarize
                """
                summary_result = openai_wrapper(
                    messages=history,
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
                    "content": summary_result,
                    "mode": "summary"
                })

            elif args.naive_summary == "10-last":
                """
                Keep 10-last history
                """
                summary = history[-11:-1]

            return summary, summary, summary, self.player_id
        
        elif mode == "discuss_on_team":
            """
            Discuss
            """
            resp = openai_wrapper(
                messages=history,
                temperature=0.1,
                **self.api_args
            )
            resp = resp["choices"][0]["message"]["content"]
            result = resp
            return result, summary, result, self.player_id
        
        elif mode == "system":
            return "Okay", summary, "Okay", self.player_id
        
        elif mode == "choose_quest_team_discussion":
            resp = openai_wrapper(
                messages=history,
                temperature=0.1,
                **self.api_args
            )
            resp = resp["choices"][0]["message"]["content"]
            result = resp

            return result, summary, result, self.player_id
        elif mode == "get_believed_sides":
            return naive_result, summary, naive_result, self.player_id

        else:
            raise NotImplementedError(
                f"There should not be other situations: {mode}."
            )