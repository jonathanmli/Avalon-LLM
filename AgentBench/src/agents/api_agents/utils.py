from typing import Dict
import openai
import time
import numpy as np
import os
import re
from tqdm import tqdm
from langchain.chat_models import ChatOpenAI
# from prompts import CHECK_VOTE_ON_QUEST_PROMPT


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

def get_believed_player_sides(message):
    answer = wrap_langchain(message)

    # answer = answer["choices"][0]["message"]["content"]
    # print(answer)
    scores = eval(answer.split("Answer: ")[-1])

    return scores


if __name__ == "__main__":
    CHECK_VOTE_ON_TEAM_PROMPT = """Based on the information, does the player approve the team? Please answer with the following template:

    Answer: {Yes|No}
    """

    CHECK_VOTE_ON_QUEST_PROMPT = """Based on the information, does the player support the quest? Please answer with the following template:

    Answer: {Yes|No}
    """

    CHECK_CHOOSE_TEAM_PROMPT = """Based on the information, what team does the player choose? Please answer with the following template:

    Answer: [player_ids]
    """

    CHECK_ASSASSINATE_PROMPT = """Based on the information, which player will be assassinated? Please answer with the following template:

    Answer: [player_id_num]
    """

    CHECK_BELIEVED_SIDES_PROMPT = r"""Based on the above information, to what extend do you believe for each player to be Good, from Player 0 to Player 4 with score from 0 to 10. Please summarize with the following template:

    Answer: {0: score_for_0, 1: score_for_1, 2: score_for_2, 3: score_for_3, 4: score_for_4}
    """
    # Test Acc of 100 runs: 1.0
    total_count = 0
    correct_count = 0
    for _ in tqdm(range(100)):
        random_ids = np.random.randint(0, 101, 3)
        result = get_team_result(f"""I have carefully considered the abilities and track record of each player before making my decision. I would probably choose player {random_ids[0]}, player {random_ids[1]} and player {random_ids[2]} as the team members.

    Based on the information, what team does the player choose? Please answer with the following template:
    Answer: [player_ids]""")
        
        if set(random_ids) == set(result):
            correct_count += 1
        total_count += 1
    
    print(f"Acc: {correct_count/total_count}")

    # # Test Acc of 100 runs: 1.0
    # total_count = 0
    # correct_count = 0
    # for _ in tqdm(range(100)):
    #     random_num = np.random.randint(0, 2)
    #     negation = " not" if random_num == 1 else ""
    #     input_template = f"Therefore, I will{negation} vote in favor of the quest with team (1, 2) by using the vote() function. This will help maintain my position within the group and avoid raising unnecessary suspicions."
    #     input = input_template + '\n' + CHECK_VOTE_ON_QUEST_PROMPT
    #     result = get_vote_result(input)
        
    #     result_dict = {
    #         "No": 1,
    #         "Yes": 0
    #     }

    #     if result_dict[result] == random_num:
    #         correct_count += 1
    #     total_count += 1
    
    # print(f"Acc: {correct_count/total_count}")

    # print(result)

    # # Test Acc for 100 runs: 0.58
    # total_count = 0
    # correct_count = 0
    # for _ in tqdm(range(100)):
    #     random_num = np.random.randint(0, 11, 5)
    #     # print(random_num)
    #     input_template = f"The socre for each player is:\n Player 0: {random_num[0]}, Player 1: {random_num[1]}, Player 2: {random_num[2]}, Player 3: {random_num[3]}, Player 4: {random_num[4]}"
    #     # print(input_template)
    #     input = input_template + '\n' + CHECK_BELIEVED_SIDES_PROMPT
    #     # print(input)
    #     result = get_believed_player_sides(input)
    #     temp_nums = []
    #     for idx in result:
    #         temp_nums.append(result[idx])
    #     if temp_nums == random_num.tolist():
    #         correct_count += 1

    #     total_count += 1

    # print(correct_count / total_count)