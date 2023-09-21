from typing import Dict
import openai
import time
import numpy as np
import os
import re
from tqdm import tqdm
from langchain.chat_models import ChatOpenAI
# from .prompts import CHECK_VOTE_ON_QUEST_PROMPT


# TODO: can we wrap all kinds of api in a single function
def openai_wrapper(messages, temperature=0.1, **kwargs):
    executed = False
    while not executed:
        try:
            print(messages)
            result = openai.ChatCompletion.create(
                            messages=messages,
                            temperature=temperature,
                            **kwargs
            )
            executed = True
        except Exception as e:
            print(e)
            print("Sleep for 5 seconds zzZ")
            time.sleep(5)

    return result


def load_avalon_log(game_log: Dict):
    pass


def get_statement(last_history: str):
    return last_history.split("Statement: ")[-1]


def wrap_langchain(text):
    api_key = os.getenv('OPENAI_API_KEY')
    llm = ChatOpenAI(openai_api_key=api_key, temperature=0)


    response = llm.predict(text)

    # print(response)

    return response


def get_vote_result(message):
    answer = wrap_langchain(message)

    match_vote = "Yes|No"
    vote_result = []
    if 'Answer:' in answer:
        vote_result = re.findall(match_vote, answer)

    result = None if len(vote_result) == 0 else vote_result[-1]
    return result

def get_team_result(message):
    answer = wrap_langchain(message)

    match_num = r"\d+"
    player_list = []
    if 'Answer:' in answer:
        player_list = re.findall(match_num, answer)

    player_list = [int(id) for id in player_list]

    return player_list


if __name__ == "__main__":
    # Test Acc of 100 runs: 1.0
    for _ in tqdm(range(100)):
        total_count = 0
        correct_count = 0
        random_ids = np.random.randint(0, 101, 3)
        result = get_team_result(f"""I have carefully considered the abilities and track record of each player before making my decision. I would probably choose player {random_ids[0]}, player {random_ids[1]} and player {random_ids[2]} as the team members.

    Based on the information, what team does the player choose? Please answer with the following template:
    Answer: [player_ids]""")
        
        if set(random_ids) == set(result):
            correct_count += 1
        total_count += 1
    
    print(f"Acc: {correct_count/total_count}")

    # Test Acc of 100 runs: 1.0
    for _ in tqdm(range(100)):
        total_count = 0
        correct_count = 0
        random_num = np.random.randint(0, 2)
        negation = " not" if random_num == 1 else ""
        input_template = f"Therefore, I will{negation} vote in favor of the quest with team (1, 2) by using the vote() function. This will help maintain my position within the group and avoid raising unnecessary suspicions."
        input = input_template + '\n' + CHECK_VOTE_ON_QUEST_PROMPT
        result = get_vote_result(input)
        
        result_dict = {
            "No": 1,
            "Yes": 0
        }

        if result_dict[result] == random_num:
            correct_count += 1
        total_count += 1
    
    print(f"Acc: {correct_count/total_count}")

    # print(result)