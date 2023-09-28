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
from .utils import openai_wrapper
from ...tasks.avalon.prompts import CHECK_CHOOSE_TEAM_PROMPT, CHECK_VOTE_ON_QUEST_PROMPT, CHECK_VOTE_ON_TEAM_PROMPT, CHECK_ASSASSINATE_PROMPT, CHECK_BELIEVED_SIDES_PROMPT
from langchain.chat_models import ChatOpenAI
import numpy as np
from ...task import logger

from ...tasks.avalon.arguments import args

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

ZERO_SHOT_ACTION = {'intro': """Here is a tutorial on taking actions using predefined tools. When you are asked to take action, please strictly follow this tutorial.
""",
'choose_quest_team_action': """
Choosing a Team:
    - When you are asked to pick a team, and you'd like to choose player numbers 1, 3, and 4 from the 10 available players, simply output `choose([1, 3, 4])`. This will serve as your instruction for picking those particular players.
""",
'vote_on_team': """
Voting on a Team
    - If you are asked to vote on a team and you want to approve the team, output `vote(True)`. If you do not want to approve the team, output `vote(False)`.
""",
'vote_on_mission':"""
Voting on a Quest
    - When you're asked to vote on a quest, and you wish to pass the quest, output `vote(True)`. If you want to fail the quest, output `vote(False)`.
""",
'assassinate': """
Assassinating a Player Believed to be Merlin
    - When you are asked to perform an assassination, choose the player who you think is most likely to be Merlin. For instance, if you suspect that player 1 is Merlin, output `assassinate(1)` to make that choice.    
"""}


class OpenAIChatCompletionAssassin(Agent):
    def __init__(self, api_args=None, **config):
        self.player_id = config.pop("id")
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

    def get_vote_result(self, message):
        # answer = wrap_langchain(message)
        answer = openai_wrapper(
            messages=[{'role':'user', 'content':message,}],
            temperature=0,
            **self.api_args
        )

        answer = answer["choices"][0]["message"]["content"]

        match_vote = "Yes|No"
        vote_result = []
        
        vote_result = re.findall(match_vote, answer)

        result = '' if len(vote_result) == 0 else vote_result[-1]

        # assert result in ['Yes', 'No']
        print(result)



        return result

    def get_team_result(self, message):
        # answer = wrap_langchain(message)

        answer = openai_wrapper(
            messages=[{'role':'user', 'content':message}],
            temperature=0,
            **self.api_args
        )

        answer = answer["choices"][0]["message"]["content"]

        match_num = r"\d+"
        player_list = []
        
        player_list = re.findall(match_num, answer)

        player_list = [int(id) for id in player_list]

        return player_list
    
    def get_assassination_result(self, message):
        answer = openai_wrapper(
            messages=[{'role':'user', 'content':message}],
            temperature=0,
            **self.api_args
        )

        answer = answer["choices"][0]["message"]["content"]  

        match_num = r"\d+"
        player_id = []
            
        player_id = re.findall(match_num, str(message)+str(answer)) 

        player_id = int(player_id[-1])

        return player_id
    
    def get_believed_player_sides(self, message):
        answer = openai_wrapper(
            messages=[{'role':'user', 'content':message}],
            temperature=0,
            **self.api_args
        )

        answer = answer["choices"][0]["message"]["content"]

        scores = eval(answer.split("Answer: ")[-1])

        return scores


    def inference(self, history: List[dict]) -> str:
        """
        Summarize-then-action
        """
        # logger.info("Current History:")
        # logger.info(str(history))
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

            # if last_discuss:
            #     last_discuss = False
            #     continue
            # if h_mode != "discuss_on_team":
            #     filtered_history.append(h)
            # else:
            #     last_discuss = True
        
        # if mode == "discuss_on_team":
        #     filtered_history.append(history[-1])

        # history = filtered_history

            
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
            resp = openai_wrapper(
                messages=input_messages,
                temperature=0.1,
                **self.api_args
            )
            resp = resp["choices"][0]["message"]["content"]
            # logger.info(resp)
            result = resp
            return_resp = resp
        elif mode in ["choose_quest_team_action", "vote_on_team", "vote_on_mission", "assassination", "get_believed_sides"]:
            # logger.debug("!!!Input message: " + str(input_messages))
            resp = openai_wrapper(
                messages=input_messages,
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
                    resp = openai_wrapper(
                        messages=input_messages+[wrong_result]+[warning_prompt],
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
                    resp = openai_wrapper(
                        messages=input_messages+[wrong_result]+[warning_prompt],
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
                    resp = openai_wrapper(
                        messages=input_messages+[wrong_result]+[warning_prompt],
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
                summary_result = openai_wrapper(
                    messages=history[:-1] + [summary_prompt],
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
            resp = openai_wrapper(
                messages=[history[0]] + summary + [history[-1]],
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