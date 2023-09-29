import anthropic
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
from .utils import claude_wrapper
from ...task import logger

from ...tasks.avalon.arguments import args
from ...tasks.avalon.prompts import CHECK_CHOOSE_TEAM_PROMPT, CHECK_VOTE_ON_QUEST_PROMPT, CHECK_VOTE_ON_TEAM_PROMPT, CHECK_ASSASSINATE_PROMPT, CHECK_BELIEVED_SIDES_PROMPT

class AvalonClaude(Agent):
    def __init__(self, api_args=None, **config):
        if not api_args:
            api_args = {}
        api_args = deepcopy(api_args)
        self.key = api_args.pop("key", None) or os.getenv('Claude_API_KEY')
        api_args["model"] = api_args.pop("model", None)
        if not self.key:
            raise ValueError("Claude API KEY is required, please assign api_args.key or set OPENAI_API_KEY environment variable.")
        if not api_args["model"]:
            raise ValueError("Claude model is required, please assign api_args.model.")
        self.api_args = api_args
        if not self.api_args.get("stop_sequences"):
            self.api_args["stop_sequences"] = [anthropic.HUMAN_PROMPT]
        super.__init__(**config)

    def inference(self, history: List[dict]) -> str:
        mode = history[-1]["mode"]
        role_name = None if "role_name" not in history[-1] else history[-1]["role_name"]
        team_size = None if "team_size" not in history[-1] else history[-1]["team_size"]
        prompt = ""
        for message in history:
            if message["role"] == "user":
                prompt += anthropic.HUMAN_PROMPT + message["content"]
            else:
                prompt += anthropic.AI_PROMPT + message["content"]
        prompt += anthropic.AI_PROMPT
        c = anthropic.Client(self.key)
        # resp = c.completion(
        #     prompt=prompt,
        #     **self.api_args
        # )
        # return resp

        summary = []
        # logger.debug("Mode: " + str(mode))
        """
        Action
        """
        if mode == 'system':
            input_messages = history
        else:
            action_prompt = {
                "role": "user",
                # "content": ZERO_SHOT_ACTION["intro"] + ZERO_SHOT_ACTION[mode] + '\n' + "Please take only one action using the functions based on the tutorial and your summary" + '\n' + history[-1]['content']
                "content": history[-1]['content']
            }
            input_messages = history[:-1] + [action_prompt]

        # print(system_prompts)
        # print(action_prompt)
        result_dict = {
            "No": 0,
            "Yes": 1
        }

        if mode == 'system':
            # logger.debug("!!!Input message: " + str(input_messages))
            resp = claude_wrapper(
                messages=input_messages,
                key=self.key,
                temperature=0.1,
                **self.api_args
            )
            resp = resp["choices"][0]["message"]["content"]
            # logger.info(resp)
            result = resp
            return_resp = resp
        elif mode in ["choose_quest_team_action", "vote_on_team", "vote_on_mission", "assassination", "get_believed_sides"]:
            # logger.debug("!!!Input message: " + str(input_messages))
            resp = claude_wrapper(
                messages=input_messages,
                key=self.key,
                temperature=0.1,
                **self.api_args
            )
            resp = resp["choices"][0]["message"]["content"]
            result = resp
            return_resp = resp
            if mode == "choose_quest_team_action":
                result = self.get_team_result(resp + '\n\n' + CHECK_CHOOSE_TEAM_PROMPT)
                if len(result) != team_size:
                    logger.warning(f"Wrong team size {len(result)}. The correct team size should be {team_size}.")
                    wrong_result = {
                        "role": "assistant",
                        "content": resp
                    }
                    warning_prompt = {
                        "role": "user",
                        "content": f"You should choose a team of size {team_size}, instead of size {len(result)} as you did."
                    }
                    resp = claude_wrapper(
                        messages=input_messages+[wrong_result]+[warning_prompt],
                        key=self.key,
                        temperature=0.1,
                        **self.api_args
                    )
                    resp = resp["choices"][0]["message"]["content"]
                    result = resp
                    return_resp = resp

                    result = self.get_team_result(resp + '\n\n' + CHECK_CHOOSE_TEAM_PROMPT)
            elif mode == "vote_on_team":
                result = self.get_vote_result(resp + '\n\n' + CHECK_VOTE_ON_TEAM_PROMPT)
                if result not in ["No", "Yes"]:
                    logger.warning(f"Error from vote on team")
                    wrong_result = {
                        "role": "assistant",
                        "content": resp
                    }
                    warning_prompt = {
                        "role": "user",
                        "content": f"You should output Yes or No to vote on the team."
                    }
                    resp = claude_wrapper(
                        messages=input_messages+[wrong_result]+[warning_prompt],
                        key=self.key,
                        temperature=0.1
                        **self.api_args
                    )
                    resp = resp["choices"][0]["message"]["content"]
                    result = resp
                    return_resp = resp

                result = result_dict[result]
            elif mode == "vote_on_mission":
                result = self.get_vote_result(resp + '\n\n' + CHECK_VOTE_ON_QUEST_PROMPT)
                if result not in ["No", "Yes"]:
                    logger.warning(f"Error from vote on mission")
                    wrong_result = {
                        "role": "assistant",
                        "content": resp
                    }
                    warning_prompt = {
                        "role": "user",
                        "content": f"You should output Yes or No to vote on the quest."
                    }
                    resp = claude_wrapper(
                        messages=input_messages+[wrong_result]+[warning_prompt],
                        key=self.key,
                        temperature=0.1
                        **self.api_args
                    )
                    resp = resp["choices"][0]["message"]["content"]
                    result = resp
                    return_resp = resp

                result = result_dict[result]
            elif mode == "assassination":
                result = self.get_assassination_result(resp + '\n\n' + CHECK_ASSASSINATE_PROMPT)
            elif mode == "get_believed_sides":
                result = []
                scores = self.get_believed_player_sides(resp + '\n\n' + CHECK_BELIEVED_SIDES_PROMPT)
                for i in range(5):
                    result.append(scores[i])
        elif mode == "summarize":
            """
            Summarize
            """
            if args.agent_summary == "full-history":
                summary_prompt = {
                    "role": "user",
                    "content": "Please summarize the history. Try to keep all the useful information, including your identification and your observations of the game."
                }
                summary_result = claude_wrapper(
                    messages=history[:-1] + [summary_prompt],
                    key=self.key,
                    temperature=0.1,
                    **self.api_args
                )
                summary_result = summary_result["choices"][0]["message"]["content"]
                # summary.append({
                #     "role": "user",
                #     "content": "Summary of previous information",
                # })
                summary.append({
                    "role": "assistant",
                    "content": "Summary of previous information:\n" + summary_result,
                })
            elif args.agent_summary == "10-last":
                summary = history[-11:-1]
            else:
                raise NotImplementedError(
                    "Value for `agent_summary` should be either `full-history` or `10-last`"
                )
            result = summary
            return_resp = summary
        elif mode == "discuss_on_team" or mode == "choose_quest_team_discussion":
            resp = claude_wrapper(
                messages=[history[0]] + summary + [history[-1]],
                key=self.key,
                temperature=0.1,
                **self.api_args
            )
        # else:

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
            return_resp = resp
        print(result)

        # time.sleep(5)


        return result, summary, return_resp, self.player_id
