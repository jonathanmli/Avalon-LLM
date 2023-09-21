from utils import openai_wrapper, get_team_result, get_vote_result
import numpy as np

if __name__ == "__main__":
    random_ids = np.random.randint(0, 101, 3)
    result = get_team_result(f"""As player 1, I have carefully considered the abilities and track record of each player before making my decision. I would probably choose player {random_ids[0]}, player {random_ids[1]} and player {random_ids[2]} as the team members.

    Based on the information, what team does the player choose? Please answer with the following template:
    Answer: [player_ids]""")
    print(result)