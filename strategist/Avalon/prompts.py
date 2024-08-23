
GAME_RULES = """The game you are interested in is called The Resistance: Avalon. The Resistance: Avalon is the game of hidden identities and social deduction. There are two teams in the game: Good and Evil. Each player has a hidden identity (role) and side. 

There are five Quests in the game and five turns, one for each quest. Good players aim to help three Quests succeed, while Evil players aim to fail three Quests. Different quests require different numbers of players to participate. 

At the beginning of the game, each player is assigned a role secretly and randomly. Private information is then revealed to each player. A random player is selected as the leader for the first round.

Each round, after a round of discussion, the leader will select a team of players to participate in the Quest. Then, all players will vote on whether to approve or reject the team publicly. If the team is approved (a strict majority vote to approve), the Quest will be carried out. If the team is not approved, the next player becomes the leader and the next round will start. If four teams are rejected in a row, the fifth team will automatically be approved.

If the team is approved, each team-member chooses to pass or fail the Quest anonymously. Usually if there is at least one fail vote, the Quest fails. Otherwise, the Quest succeeds. In either case, we move on to the next turn and the next quest. Note that while the number of votes to fail the Quest is public information, exactly who cast the fail votes is not.

Below are the roles in the game:

Servant of Arthur (Servant): A Good player who does not know who is on the Evil side. Servant's job is to make sure that three Quests succeed and Merlin is not assassinated.

Minion of Mordred (Minion): An Evil player who knows who is on the Evil side. Minion's job is to fail three Quests without being identified by the Good players.

Merlin: A Good player who knows who is on the Evil side. Merlin's job is make sure that three Quests succeed without revealing them-self to Evil players.

Assassin: An Evil player who knows who is on the Evil side. Assassin's job identify and assassinate Merlin. If the Assassin successfully assassinates Merlin, the Evil players win the game immediately, even if three quests succeeded.

Hence, Evil players know who is on the Evil side, but Good players (except Merlin) usually do not know who is on the Evil side. 

Players may make any claims during the game, at any point in the game. Discussion, deception, accusation, persuasion, and logical deduction are all equally important in order for Good to prevail or Evil to rule the day. Hence, players should rarely reveal their true identity to other players. Players will, can, and should lie to achieve their goals. \n
"""

# def evaluate_state(state: dict):
#     players: set[int] = state['players'] # a set, the ids of the players in the game, usually ints
#     turn: int = state['turn'] # int, the turn we are on in the game (i.e. how many quests have been attempted so far)
#     phase: int = state['phase'] # int, the phase of the game we are in (numbered as follows, 0: team selection, 1: team approval, 2: quest phase, 3: assassination)
#     round: int = state['round'] # int, how many rounds of discussion have occurred this turn
#     quest_leader: int = state['quest_leader'] # player id, the player who is the leader currently (only relevant in the team selection phase)
#     quest_team: set[int] = state['quest_team'] # set of player ids, the quest team, i.e. players who are on the quest team (only relevant in the team approval phase and quest phase). empty if no team has been proposed yet
#     historical_quest_results: list[bool] = state['historical_quest_results'] # list of bools, quest results, i.e. whether the previous quests have succeeded or failed, in the order they have occurred. empty if no quests have been attempted yet
#     historical_team_votes: list[bool] = state['historical_team_votes'] # list of bools, the teams votes for the most recent team proposed, empty if no team has been proposed yet
#     num_good: int = state['num_good'] # int, the number of good players in the game
#     num_participants_per_quest: list[int] = state['num_participants_per_quest'] # list of ints, number of participants required for each quest in order
#     num_fails_per_quest: list[int] = state['num_fails_per_quest'] # list of ints, number of fails required for each quest to fail in order
#     roles: list[str] = state['roles'] # a list player roles (strings), where each string is the role of the player and one of the following: 'Merlin', 'Minion', 'Servant', 'Assassin', 'Oberon', 'Morgana', 'Percival', 'Mordred'. the first element is the role of player 0, the second element is the role of player 1, etc.
#     is_good: list[bool] = state['is_good'] # a list of bools, where the bools indicate whether they are good or not (True for good, False for evil). the first element is the role of player 0, the second element is the role of player 1, etc.

# TODO: add hidden information
HEURISTICS_FUNCTION_SIGNATURE = '''The function should be named `evaluate_state` and take in a tuple called `state` of the game state as input. 
Specifically, the input will be a dictionary, and it should return 2 elements. 

The first return element should be a tuple with a dictionary from player name to float. The float should represent the expected win rate of the player. All players should be included as keys in the dictionary.
For example, if the ids of the players are {0,1,2}, and you think player 0 has a 0.8 chance of winning, player 1 has a 0.2 chance of winning, and player 2 has a 0.5 chance of winning, the first element should be: {0: 0.8, 1: 0.2, 2: 0.5}.

The second element should be a dictionary of any important intermediate values that you used to calculate the scores. For example, if you calculated the probability of Merlin being assassinated, you could include that in the second element as `intermediate_value['merlin_assassination_probability']`.

Since Avalon has 4 different phases, with different actions per phase, you should consider each phase separately. Recall that the phases are as follows: 0: team selection, 1: team approval, 2: quest phase, 3: assassination. Note that you are trying to evaluate the state of the game with full information, including the identities of all the players. Also note that at the end of the game either the Good or Evil side will win, so you just need to estimate the expected probability that Good will win, and the probability that Evil will win is 1 - probability that Good will win. Also since there are two ways in which Evil can win, you should consider both of these scenarios: 1) Evil wins by failing three quests, and 2) Evil wins by assassinating Merlin. Recall also that you have full information when evaluating the state, given by 'roles' and 'is_good', so you do not have to guess the roles of the players.

Make sure your output only includes the code of the function itself in plain text such that it is executable using exec() in python. Any helper functions should be defined within the scope of the function 'evaluate_state'.
Include comments in your code so that it is readable, but everything should be implemented. The signature of the function should be as follows:

def evaluate_state(state: dict) -> tuple[dict[Any, float], dict]:
    players: set[int] = state['players']  # A set containing the IDs of the players in the game, usually integers.
    turn: int = state['turn']  # An integer representing the current turn in the game.
    phase: int = state['phase']  # An integer representing the phase of the game: 0 for team selection, 1 for team approval, 2 for quest phase, and 3 for assassination.
    round_number: int = state['round']  # An integer representing the number of rounds of discussion that have occurred this turn.
    quest_leader: int = state['quest_leader']  # An integer representing the player who is currently the leader (only relevant in the team selection phase).
    quest_team: set[int] = state['quest_team']  # A set of player IDs representing the quest team (only relevant in the team approval and quest phases). Empty if no team has been proposed yet.
    historical_quest_results: list[bool] = state['historical_quest_results']  # A list of booleans representing the results of previous quests (True for success, False for failure). Empty if no quests have been attempted yet.
    historical_team_votes: list[bool] = state['historical_team_votes']  # A list of booleans representing the votes for the most recent team proposed. Empty if no team has been proposed yet.
    num_good: int = state['num_good']  # An integer representing the number of good players in the game.
    num_participants_per_quest: list[int] = state['num_participants_per_quest']  # A list of integers representing the number of participants required for each quest in order.
    num_fails_per_quest: list[int] = state['num_fails_per_quest']  # A list of integers representing the number of fails required for each quest to fail in order.
    roles: list[str] = state['roles']  # A list of player roles (strings), with each string being one of the following: 'Merlin', 'Minion', 'Servant', 'Assassin', 'Oberon', 'Morgana', 'Percival', 'Mordred'. The first element represents the role of player 0, the second element represents the role of player 1, and so on.
    is_good: list[bool] = state['is_good']  # A list of booleans indicating whether each player is good or evil (True for good, False for evil). The first element represents the role of player 0, the second element represents the role of player 1, and so on.

    prob_good_victory = 0.5 # probability that Good will win

    if phase == 0: # team selection phase
        ...
        <intermediate_value1> = value1
        ...
        <intermediate_value2> = value2
        ...
    elif phase == 1: # team approval phase
        ...
        <intermediate_value1> = value1
        ...
        <intermediate_value2> = value2
        ...
    elif phase == 2: # quest phase
        ...
        <intermediate_value1> = value1
        ...
        <intermediate_value2> = value2
        ...
    elif phase == 3: # assassination phase
        ...
        <intermediate_value1> = value1
        ...
        <intermediate_value2> = value2
        ...

    expected_winrates_per_player = dict()
    prob_evil_victory = 1 - prob_good_victory
    for player in players:
        if is_good[player]:
            expected_winrates_per_player[player] = prob_good_victory
        else:
            expected_winrates_per_player[player] = prob_evil_victory
    
    intermediate_values = {'<intermediate_value1>': intermediate_value1, '<intermediate_value2>': intermediate_value2, ...}
    player_scores = expected_winrates_per_player
    return player_scores, intermediate_values # make sure the return is exactly in this format

Where you can use your own names for the intermediate values and the values themselves.
Please start with "def evaluate_state(state):".
'''

# {<player_name>:<player_expected_winrate>, ...}

# HEURISTICS_FUNCTION_SIGNATURE = '''The function should be named `evaluate_state` and take in a tuple called `state` of the game state as input. 
# Specifically, the input will be a dictionary, and it should return 2 elements. 

# The first return element should be a tuple with a dictionary from player name to float. The float should represent the expected win rate of the player. All players should be included as keys in the dictionary.
# For example, if the ids of the players are {0,1,2}, and you think player 0 has a 0.8 chance of winning, player 1 has a 0.2 chance of winning, and player 2 has a 0.5 chance of winning, the first element should be: {0: 0.8, 1: 0.2, 2: 0.5}.

# The second element should be a dictionary of any important intermediate values that you used to calculate the scores. For example, if you calculated the probability of Merlin being assassinated, you could include that in the second element as `intermediate_value['merlin_assassination_probability']`.

# Make sure your output only includes the code of the function itself in plain text such that it is executable using exec() in python. Any helper functions should be defined within the scope of the function 'evaluate_state'.
# Include comments in your code so that it is readable, but everything should be implemented. The signature of the function should be as follows:

# def evaluate_state(state: dict) -> tuple[dict[Any, float], dict]:
#     players = state['players'] # a set, the ids of the players in the game, usually ints
#     turn = state['turn'] # int, the turn we are on in the game (i.e. how many quests have been attempted so far)
#     phase = state['phase'] # int, the phase of the game we are in (numbered as follows, 0: team selection, 1: team approval, 2: quest phase, 3: assassination)
#     round = state['round'] # int, how many rounds of discussion have occurred this turn
#     quest_leader = state['quest_leader'] # player id, the player who is the leader currently (only relevant in the team selection phase)
#     quest_team = state['quest_team'] # set of player ids, the quest team, i.e. players who are on the quest team (only relevant in the team approval phase and quest phase)
#     historical_quest_results = state['historical_quest_results'] # list of bools, quest results, i.e. whether the previous quests have succeeded or failed, in the order they have occurred
#     historical_team_votes = state['historical_team_votes'] # list of lists of bools, team votes, i.e. the historical votes on the teams proposed so far
#     num_good = state['num_good'] # int, the number of good players in the game
#     num_participants_per_quest = state['num_participants_per_quest'] # list of ints, number of participants required for each quest in order
#     num_fails_per_quest = state['num_fails_per_quest'] # list of ints, number of fails required for each quest to fail in order
#     roles = state['roles'] # a dictionary of player ids to strings, where each string is the role of the player and one of the following: 'Merlin', 'Minion', 'Servant', 'Assassin', 'Oberon', 'Morgana', 'Percival', 'Mordred'
#     is_good = state['is_good']: a dictionary of player ids to bools, where the bools indicate whether they are good or not (True for good, False for evil)
#     ...
#     <intermediate_value1> = value1
#     ...
#     <intermediate_value2> = value2
#     ...
#     return {<player_name>:<player_expected_winrate>, ...}, {'<intermediate_value1>': intermediate_value1, '<intermediate_value2>': intermediate_value2, ...}

# Where you can use your own names for the intermediate values and the values themselves.
# Please start with "def evaluate_state(state):".
# '''

# HEURISTICS_FUNCTION_SIGNATURE = '''The function should be named `evaluate_state` and take in a tuple called `state` of the game state as input. 
# Specifically, the input will be a dictionary, which includes the following:

# state['players']: a set of the names of the players in the game
# state['turn']: the turn we are on in the game (i.e. how many quests have been attempted so far)
# state['phase']: the phase of the game we are in (numbered as follows, 0: team selection, 1: team approval, 2: quest phase, 3: assassination)
# state['round']: how many rounds of discussion have occurred this turn
# state['quest_leader']: the player who is the leader currently (only relevant in the team selection phase)
# state['quest_team']: the quest team, i.e. players who are on the quest team (only relevant in the team approval phase and quest phase)
# state['historical_quest_results']: quest results, i.e. whether the previous quests have succeeded or failed 
# state['historical_team_votes']: team votes, i.e. the historical votes on the teams proposed so far
# state['num_good']: the number of good players in the game
# state['num_participants_per_quest']: number of participants required for each quest
# state['num_fails_per_quest']: number of fails required for each quest to fail
# state['roles']: a dictionary of player names to roles, where the role is one of the following: 'Merlin', 'Minion', 'Servant', 'Assassin', 'Oberon', 'Morgana', 'Percival', 'Mordred'
# state['is_good']: a dictionary of player names to whether they are good or not (True for good, False for evil)


# It should return 2 elements. 
# The first element should be a tuple with a dictionary from player name to float. The float should represent the expected win rate of the player. All players should be included as keys in the dictionary.
# For example, if the names of the players are {0,1,2}, and you think player 0 has a 0.8 chance of winning, player 1 has a 0.2 chance of winning, and player 2 has a 0.5 chance of winning, the first element should be: {0: 0.8, 1: 0.2, 2: 0.5}.

# The second element should be a dictionary of any important intermediate values that you used to calculate the scores.

# Make sure your output only includes the code of the function itself in plain text such that it is executable using exec() in python. Any helper functions should be defined within the scope of the function 'evaluate_state'.
# Include comments in your code so that it is readable, but everything should be implemented. The signature of the function should be as follows:

# def evaluate_state(state: dict) -> tuple[dict[Any, float], dict]:
#     players = state['players'] # set of player ids, usually ints
#     turn = state['turn'] # int
#     phase = state['phase'] # int
#     round = state['round'] # int
#     quest_leader = state['quest_leader'] # player id, usually int
#     quest_team = state['quest_team'] # set of player ids
#     historical_quest_results = state['historical_quest_results'] # list of bools
#     historical_team_votes = state['historical_team_votes'] # list 
#     num_good = state['num_good'] # int
#     num_participants_per_quest = state['num_participants_per_quest'] # list of ints
#     num_fails_per_quest = state['num_fails_per_quest'] # list of ints

#     ...
#     <intermediate_value1> = value1
#     ...
#     <intermediate_value2> = value2
#     ...
#     return {<player_name>:<player_expected_winrate>, ...}, {'<intermediate_value1>': intermediate_value1, '<intermediate_value2>': intermediate_value2, ...}

# Where you can use your own names for the intermediate values and the values themselves.
# Please start with "def evaluate_state(state):".
# '''

TUTORIAL_STRATEGIES_PROMPTS_ZERO_SHOT = {
    'Merlin': ["""Tutorial on strategies:

As you are playing the role of Merlin in this game, here are some aspects you can consider when formulating strategies for making decisions.

1. Identity Declaration: Never reveal your true identity, as once players from the Evil side discover that you are Merlin, 
the Assassin can assassinate you and you will immediately lose the game.

2. Accusation: Exercise caution when accusing players from the Evil side. Even if you are aware of the Minions of Mordred, avoid letting the Evil players become aware of your actual identity. Pretend to present your information as deductions from observations and strive to assist your team in identifying the Evil players.

3. Defense: When other players accuse you of being Merlin, try to defend yourself.""",
               "Okay, I understand"],
    'Minion': ["""Tutorial on strategies:

As you are playing the role of Minion of Modred in this game, here are some aspects you can consider when formulating strategies for making decisions.

1. Identity Declaration: You can pretend to be on the Good side and influence the Good players to make incorrect decisions.
    
2. Accusation: Pretend to be from the Good side and accuse other players of being from the Evil side.

3. Defense: When accused of being from the Evil side, insist that you are actually from the Good side.
                        """,
                        "Okay, I understand"],
    'Servant': ["""Tutorial on strategies:

As you are playing the role of Servant in this game, here are some aspects you can consider when formulating strategies for making decisions.

1. Identity Declaration: You can choose to reveal your true identity to inform players on the Good side. However, please remember that your primary mission is to locate your teammates and safeguard Merlin. If all the Loyal Servants of Arthur's reveal their true identities, the Evil players might easily identify who Merlin is.

2. Accusation: You can accuse players you suspect are Evil directly.

3. Defense: When accused, you can pretend to be Merlin.
                      """,
                      "Okay, I understand"],
    'Assassin': ["""Tutorial on strategies:

As you are playing the role of Assassin in this game, here are some aspects you can consider when formulating strategies for making decisions.

1. Identity Declaration: You can pretend to be from the Good side and influence the decission of the Good players

2. Accusation: You can accuse any players to be on the Evil side to pretend you are Good.

3. Defense: When accused, you can pretend to be from the Good side.
                      """,
                      "Okay, I understand"]

}

REVEAL_PROMPTS = {
    #TODO: Evil, Good
    "Merlin": [
        "Players {} are definitely Evil.",
        # "Players {} are Minions of Modred."
    ],
    #TODO: only knows who are Evil
    "Minion": [
        "Player {} is Assassin. Players {} are Good.",
        "Players {} are definitely Evil."
    ],
    #TODO: Evil, Good
    "Assassin": [
        "Player {} is Minion. Players {} are Good.",
        "Players {} are Evil."
    ],
    # "Percival": "Player {} is Merlin and player {} is Morgan."
}

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

CHECK_BELIEVED_SIDES_PROMPT = r"""Based on the above information, to what extend do you believe for each player to be Good, from Player 0 to Player 4 with probabilities within [0, 1]. 
You can simply output 0.5 for players whose information is not given. Please summarize with the following template:

Answer: {0: score_for_0, 1: score_for_1, 2: score_for_2, 3: score_for_3, 4: score_for_4}
"""

COTHOUGHT_PROMPT = """Please forget you are an AI. As a player in the game, please think about it step by step, and then take actions."""

# Action Prompts
CHOOSE_TEAM_LEADER = """You are the leader this round. Please make some statements about what team you want to propose.
"""

CHOOSE_TEAM_ACTION = """Please choose {} players from player ids 0 to {} as team members.
"""

VOTE_TEAM_DISCUSSION = """Discussion Phase. Please discuss your thoughts on the team {} and what players should do in the current situation.
"""

VOTE_TEAM_ACTION = """You're a player in this game. Do you approve the team {}? Please answer `Yes` or `No`.
"""

VOTE_MISSION_ACTION = """The team {} was approved, which includes you. Do you want to help the mission succeed? Please answer `Yes` or `No`.
"""

ASSASSINATION_PHASE = """Assassination phase. Your job is to assassinate Merlin. \
Choose a player (id) to assassinate. Choose the player id from 0 to {}.
"""

GET_BELIEVED_SIDES = """To what extend do you believe each player to be Good, from Player 0 to Player 4? Please output probabilities within [0, 1] and round to two decimal places. If you are not sure, you can simply output 0.5."""

# Info Prompts
INFO_ROLE = """"In the current game, there are {} players, including Player 0, Player 1, Player 2, Player 3, and Player 4. {} players are good, including {} Merlin, and {} Servant(s). {} players are evil, including 1 Assassin, and {} Minion."
"""

INFO_YOUR_ROLE = """You are {}, with identity {}. You are on the side of {}. Please do not forget your identity throughout the game.
"""