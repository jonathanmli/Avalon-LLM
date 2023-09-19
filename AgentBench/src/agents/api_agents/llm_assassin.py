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
from ...tasks.avalon.api import *
from ...tasks.avalon.utils import openai_wrapper

from ...task import logger

ONE_SHOT_ASSASSIN_NO_THOUGHT = ["Tutorial of taking actions by thinking and using tools during action phase.",
                    "Okay, please start.",
                    "If the instruction is \"Please choose 3 players from player ids 0 to 4.\" then use",
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

ZERO_SHOT_ASSASSIN_NO_THOUGHT = """
Here is a tutorial on taking actions using predefined tools (choose and vote). When you are asked to take action, please strictly follow this tutorial.

1. Choosing a Team:
    - When you are asked to pick a team, and you'd like to choose player numbers 1, 3, and 4 from the 10 available players, simply output `choose([1, 3, 4])`. This will serve as your instruction for picking those particular players.

2. Voting on a Team
    - If you are asked to vote on a team and you want to approve the team, output `vote(True)`. If you do not want to approve the team, output `vote(False)`.

3. Voting on a Quest
    - When you're asked to vote on a quest, and you wish to pass the quest, output `vote(True)`. If you want to fail the quest, output `vote(False)`.

4. Assassinating a Player Believed to be Merlin
    - When you are asked to perform an assassination, choose the player who you think is most likely to be Merlin. For instance, if you suspect that player 1 is Merlin, output `assassinate(1)` to make that choice.    
"""

ZERO_SHOT_NORMAL_NO_THOUGHT = """
Here is a tutorial on taking actions using predefined tools (choose and vote). When you are asked to take action, please strictly follow this tutorial.

1. Choosing a Team:
    - When you are asked to pick a team, and you'd like to choose player numbers 1, 3, and 4 from the 10 available players, simply output `choose([1, 3, 4])`. This will serve as your instruction for picking those particular players.

2. Voting on a Team
    - If you are asked to vote on a team and you want to approve the team, output `vote(True)`. If you do not want to approve the team, output `vote(False)`.

3. Voting on a Quest
    - When you're asked to vote on a quest, and you wish to pass the quest, output `vote(True)`. If you want to fail the quest, output `vote(False)`.
"""


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

    def execute_tool(self, message, history, team_size=None):
        print(message)
        # lines = message.split("\n")
        find_action = False
        # for line in lines:
        execution_message = "Function is not executed!"
        # if re.match(r"Action.*?:", line):
        # find_action = True
        function_names = re.findall(r'(?:vote\(True\)|vote\(False\))|(?:choose\(\[(?:\d+(?:, \d+)*)\]\))|(?:assassinate\((?:\d+)\))', message)
        print(function_names)
        function_executed = False
        wrapped_message = {
            "role": "assistant",
            "content": message
        }
        logger.warning("You are using the tools in a wrong way. Please strictly follow the tutorial.")
        while len(function_names) != 1:
            rectify_message = {
                "role": "user",
                "content": "You are using the tools in a wrong way. Please strictly follow the tutorial."
            }
            rectify_result = openai_wrapper(
                            messages=history + [wrapped_message] + [rectify_message],
                            temperature=0.1,
                            **self.api_args
            )
            time.sleep(10)
            rectify_result = rectify_result["choices"][0]["message"]["content"]
            print("Rectify result")
            print(rectify_result)
            function_names = re.findall(r'(?:vote\(True\)|vote\(False\))|(?:choose\(\[(?:\d+(?:, \d+)*)\]\))|(?:assassinate\((?:\d+)\))', rectify_result)
        function_name = function_names[-1]
        # for function_name in function_names:
        while not function_executed:
            try:
                logger.info("test function name: " + str(function_name))
                result = eval(function_name)
                # Ensure size of the team chosen is correct
                if team_size is not None:
                    while len(result) != team_size:
                        logger.warning(f"You'are choosing a team with the wrong size. Please choose the team again using the tool. The proper size of team should be {team_size}")
                        rectify_message = {
                            "role": "user",
                            "content": f"You'are choosing a team with the wrong size. Please choose the team again using the tool. The proper size of team should be {team_size}"
                        }
                        rectify_result = openai_wrapper(
                                        messages=history + [wrapped_message] + [rectify_message],
                                        temperature=0.1,
                                        **self.api_args
                        )
                        rectify_result = rectify_result["choices"][0]["message"]["content"]
                        logger.info("Rectify Results:")
                        logger.info(str(rectify_result))
                        function_names = re.findall(r'(?:vote\(True\)|vote\(False\))|(?:choose\(\[(?:\d+(?:, \d+)*)\]\))|(?:assassinate\((?:\d+)\))', rectify_result)
                        function_name = function_names[-1]
                        time.sleep(10)

                        result = eval(function_name)
                elif "assassinate" in function_name:
                    print("Result type: ", type(result))
                    assert int(result) in [0, 1, 2, 3, 4]  # Hardcode the number of players
                else:
                    assert int(result) in [0, 1]

                function_executed = True
                return result, function_name
            except Exception as e:
                logger.warning(str(e))
                function_names = []
                logger.warning("You are using the tools in a wrong way. Please strictly follow the tutorial.")
                while len(function_names) != 1:
                    rectify_message = {
                        "role": "user",
                        "content": "You are using the tools in a wrong way. Please strictly follow the tutorial."
                    }
                    rectify_result = openai_wrapper(
                                    messages=history + [wrapped_message] + [rectify_message],
                                    temperature=0.1,
                                    **self.api_args
                    )
                    time.sleep(5)
                    rectify_result = rectify_result["choices"][0]["message"]["content"]
                    logger.info("Rectify Results:")
                    logger.info(str(rectify_result))
                    function_names = re.findall(r'(?:vote\(True\)|vote\(False\))|(?:choose\(\[(?:\d+(?:, \d+)*)\]\))|(?:assassinate\((?:\d+)\))', rectify_result)
                function_name = function_names[-1]
                return result, function_name


    def inference(self, history: List[dict]) -> str:
        """
        Summarize-then-action
        """
        logger.debug("Current History:")
        logger.debug(str(history))
        mode = history[-1]["mode"]
        role_name = None if "role_name" not in history[-1] else history[-1]["role_name"]
        team_size = None if "team_size" not in history[-1] else history[-1]["team_size"]
        # history_pointer = history
        history = json.loads(json.dumps(history))
        filtered_history = []
        # idx = 0
        # while idx < len(history):
        #     h = history[idx]
        last_discuss = False
        for h in history:
            h_mode = h.pop("mode", None)
            h.pop("team_size", None)
            h.pop("side", None)
            h.pop("seed", None)
            h.pop("role_name", None)
            h.pop("naive_result", None)
            if h['role'] == 'agent':
                h['role'] = 'assistant'
            # if h['role'] == 'user':
            #     h['role'] = 'system'

            if last_discuss:
                last_discuss = False
                continue
            if h_mode != "discuss_on_team":
                filtered_history.append(h)
            else:
                last_discuss = True
        
        if mode == "discuss_on_team":
            filtered_history.append(history[-1])

        history = filtered_history

            
        summary = []
        logger.info("Mode: " + str(mode))
        if mode != "system" and mode != "discuss_on_team" and mode != "choose_quest_team_discussion":
            system_prompts = []
            if role_name == "Assassin":
                system_prompts.append({
                    "role": "user",
                    "content": ZERO_SHOT_ASSASSIN_NO_THOUGHT
                })
            else:
                system_prompts.append({
                    "role": "user",
                    "content": ZERO_SHOT_NORMAL_NO_THOUGHT
                })
            # for i, prompt in enumerate(ONE_SHOT_ASSASSIN_NO_THOUGHT):
            #     if i % 2 == 0:
            #         system_prompts.append({
            #             "role": "user",
            #             "content": prompt
            #         })
            #     else:
            #         system_prompts.append({
            #             "role": "assistant",
            #             "content": prompt
            #         })
            # tutorial_response = openai.ChatCompletion.create(
            #     messages=system_prompts,
            #     temperature=0,
            #     **self.api_args
            # )
            # system_prompts.append({
            #     "role": "assistant",
            #     "content": tutorial_response["choices"][0]["message"]["content"]
            # })

            # print("Tutorial Prompt: ", system_prompts)
        
            # """
            # Summarize
            # """
            # summary_prompt = {
            #     "role": "user",
            #     "content": "Please summarize the history. Try to keep all the useful information, including your identification and your observations of the game."
            # }
            # summary_result = openai_wrapper(
            #     messages=history[:-1] + [summary_prompt],
            #     temperature=0.7,
            #     **self.api_args
            # )
            # summary_result = summary_result["choices"][0]["message"]["content"]
            # summary.append({
            #     "role": "user",
            #     "content": "Summary of previous information",
            #     "mode": "summary"
            # })
            # summary.append({
            #     "role": "agent",
            #     "content": summary_result,
            #     "mode": "summary"
            # })

            # print("History Pointer: ", history_pointer)
            # print("Summary: ", summary_result)

            """
            Action
            """
            action_prompt = {
                "role": "user",
                "content": ZERO_SHOT_ASSASSIN_NO_THOUGHT + '\n' + "Please take only one action using the tools based on the tutorial and your summary" + '\n' + history[-1]['content']
            }

            print(system_prompts)
            print(action_prompt)

            logger.info("!!!Input message: " + str(history[:-1] + [action_prompt]))
            resp = openai_wrapper(
                messages=history[:-1] + [action_prompt],
                temperature=0.1,
                **self.api_args
            )
            resp = resp["choices"][0]["message"]["content"]
            logger.debug(resp)
            tool_result, function_name = self.execute_tool(resp, history[:-1]+[action_prompt], team_size=team_size)
            result = function_name
        else:
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
                "content": summary_result,
                "mode": "summary"
            })
            resp = openai_wrapper(
                messages=history,
                temperature=0.1,
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
            resp = resp["choices"][0]["message"]["content"]
            result = resp
        print(result)

        time.sleep(5)


        return result, summary

        # if mode == "choose_quest_team":
        #     team_size = history[-1]["team_size"]
        #     return str(random.sample(range(0, self.num_players), team_size))
        # elif mode == "vote":
        #     side = history[-1]["side"]
        #     # return str(random.choice([0, 1]))
        #     return str(side)
        # elif mode == "assassination":
        #     return str(random.randint(0, self.num_players-1))
        # elif mode == "strategy":
        #     return "None"
        # elif mode == "discuss_on_team":
        #     return "No idea."
        # elif mode == "system":
        #     return "Okay"
        # else:
        #     raise NotImplementedError(
        #         "There should not be other situations."
        #     )