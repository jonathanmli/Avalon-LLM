from strategist.GOPS.baseline_models_GOPS import GOPSState2

HIDDEN_STATE_ENUMERATOR_PROMPT = '''Recall the rules of the game and enumerate the possible hidden states of the game that we could be in. Write down your thoughts and output the list of states

Thought:
your thoughts here

States:
gather your thoughts in a python list, for example: [your thought 1, your thought 2, your thought 3]
'''
HIDDEN_STATE_PREDICTOR_PROMPT = b'''Given the current situation, for each of the hidden states, what is the probability of being in that state? Write down your thoughts and output the dict of probabilities

Thought:
your thoughts here

Probabilities:
gather your thoughts in a python dict, for example: {state1: prob1, state2: prob2, state3: prob3}'''
HIDDEN_STATE_PREDICTOR_PROMPT_SINGLE = '''Given the current situation, on a scale of 0.0 to 1.0, what is the likelihood of being this hidden state?'''

FORWARD_ENUMERATOR_PROMPT = '''Recall the rules of the game and enumerate the possible next states of the game that we could be in given the current state and action taken. Write down your thoughts and output the list of states

Thought:
your thoughts here

States:
gather your thoughts in a python list, for example: [your thought 1, your thought 2, your thought 3]'''
FORWARD_PREDICTOR_PROMPT = b'''Given the current state and action taken, for each of the next states, what is the probability of being in that state? Write down your thoughts and output the dict of probabilities

Thought:
your thoughts here

Probabilities:
gather your thoughts in a python dict, for example: {state1: prob1, state2: prob2, state3: prob3}'''

VALUE_PREDICTOR_PROMPTS = ['''Given the current situation, what is the probability of winning? Write down your thoughts and output the probability.

Thought:
your thoughts here

Probability:
the probability here''', 
'''Given the current situation, what is the value of the game? Write down your thoughts and rate on a scale from 0 (no value) to 10 (extremely valuable).

Thoughts:
your thoughts here

Value:
the value here''',
'''Given the current situation, how many more points do you expect to get than your opponent at the end of the game? Write down your thoughts and output the number of points.

Thought:
your thoughts here

Your score:
the score you expect to get at the end of the game

Opponent's score:
the score you expect your opponent to get at the end of the game

Points:
the difference between your score and your opponent's score at the end of the game''', 
'''Given the current situation, how many more points do you expect to win in the future? Write down your thoughts and output the number of points.

Thought:
your thoughts here

Points:
the number of points here''',
'''Given the current situation, how many points have you won so far? Write down your thoughts and output the number of points.

Thought:
your thoughts here

Points:
the number of points here''',
]

HEURISTICS_FUNCTION_PROMPTS = ['''Given the rules of the game, come up with a function that can be used to evaluate the value of a state (ie. how many points you expected you will get and how many points you expect your opponent will get). Write down your thoughts and output the function.

Thoughts:
your thoughts here

Pseudocode: 
the pseudocode for your function here. you can be as abstract as you want.
                               
Below is an example for the game of tic-tac-toe, where I calculate the probability of winning for each player.
                               
"
Thoughts:
- If I have 3 in a row, I win
- If my opponent has 3 in a row, I lose
- Otherwise, the game is a draw
- The center position is the most important position
                                                     
Pseudocode:            
Use an if-else function to check the following,
- If it is my turn check if I can win, if so return (1.0,0.0)
- If it is my opponent's turn check if they can win, if so return (0.0,1.0)
- If I control the center or it is my turn and the center is empty, return (0.8,0.2)
- If the opponent controls the center or it is my opponent's turn and the center is empty, return (0.2,0.8)                                 
"
                 
Remember that the function should output the score that you expect to get at the end of the game and the score that you expect your opponent will get at the end of the game. For example, if you think you will win 12 total points by the end of the game and your opponent will win 8 total points, the function should return (12, 8).''']

GOPS_VALUE_FUNCTION_PROMPT = '''Implement the function you just described into python code.

The function should be named `evaluate_state` and take in a tuple called `state` of the game state as input. 
Specifically, the input tuple will be of length 7, with each element representing the following:
state[0]: a list of the score cards (integers) that have been played, in the order they were played
state[1]: a list of the cards (integers) you have played, in the order they were played
state[2]: a list of the cards (integers) your opponent has played, in the order they were played
state[3]: boolean, true if it is you and your opponent's turn to play, false if it is time to draw a new score card
state[4]: integer, your score so far
state[5]: integer, your opponent's score so far
state[6]: a list of the score cards (integers) left in the deck

It should return a tuple of 2 elements, with the first element being the score you expect you will get at the end of the game, and the second element being the score you expect your opponent will get at the end of the game.
For example, if you think you will win 12 total points by the end of the game and your opponent will win 8 total points, the function should return (12, 8).

Make sure your output only includes the code of the function itself in plain text such that it is executable using exec() in python. Any helper functions should be defined within the scope of the function 'evaluate_state'.
Include comments in your code so that it is readable, but everything should be implemented. The signature of the function should be as follows:

def evaluate_state(state) -> tuple[int, int]:
    ...
    return (your_score, opponent_score)

Please start with "def evaluate_state(state):".
'''

# VALUE_FUNCTION_SIGNATURE = '''The function (written in python) should be named `evaluate_state` and take in a tuple called `state` of the game state as input. 
# Specifically, the input tuple will be of length 9, with each element representing the following:
# state[0]: a list of the score cards (integers) that have been played, in the order they were played
# state[1]: a list of the cards (integers) player 0 has played, in the order they were played
# state[2]: a list of the cards (integers) player 1 has played, in the order they were played
# state[3]: boolean, true if it is you and your opponent's turn to play, false if it is time to draw a new score card
# state[4]: float or integer, player 0's so far
# state[5]: float or integer, player 1's score so far
# state[6]: a set of the score cards (integers) left in the deck
# state[7]: a set of the cards (integers) left in player 0's hand
# state[8]: a set of the cards (integers) left in player 1's hand

# It should return 2 elements. 
# The first element should be a tuple with 2 floats: the first element being the score you expect player 0 will get at the end of the game, and the second element being the score you expect player 1 will get at the end of the game.
# The second element should be a dictionary of any important intermediate values that you used to calculate the scores.
# For example, if you think player 0 will win 12 total points by the end of the game and player 1 will win 8 total points, the function should return (12, 8).

# Make sure your output only includes the code of the function itself in plain text such that it is executable using exec() in python. Any helper functions should be defined within the scope of the function 'evaluate_state'.
# Include comments in your code so that it is readable, but everything should be implemented. The signature of the function should be as follows:

# def evaluate_state(state) -> tuple[tuple[float, float], dict]:
#     score_cards = state[0] # list, which may be of the same length as player_0_played_cards and player_1_played_cards or one more
#     player_0_played_cards = state[1] # list
#     player_1_played_cards = state[2] # list, same length as player_0_played_cards
#     is_turn = state[3] # bool
#     player_0_score = state[4] # float or int
#     player_1_score = state[5] # float or int
#     score_deck = state[6] # set, either same length as player_0_played_cards and player_1_played_cards or one less
#     player_0_hand = state[7] # set
#     player_1_hand = state[8] # set, same length as player_0_hand
#     ...
#     <intermediate_value1> = value1
#     ...
#     <intermediate_value2> = value2
#     ...
#     player_scores = (player_0_expected_score, player_1_expected_score)
#     intermediate_values = {'<intermediate_value1>': intermediate_value1, '<intermediate_value2>': intermediate_value2, ...}
#     return player_scores, intermediate_values # make sure the return is exactly in this format

# Where you can use your own names for the intermediate values and the values themselves.
# Please start with "def evaluate_state(state):"
# '''

VALUE_FUNCTION_SIGNATURE = '''The function (written in python) should be named `evaluate_state` and take in a tuple called `state` of the game state as input. 
Specifically, the input tuple will be of length 9, and it should return 2 elements. 
The first element should be a tuple with 2 floats: the first element being the score you expect player 0 will get at the end of the game, and the second element being the score you expect player 1 will get at the end of the game.
The second element should be a dictionary of any important intermediate values that you used to calculate the scores.
For example, if you think player 0 will win 12 total points by the end of the game and player 1 will win 8 total points, the function should return (12, 8).

Make sure your output only includes the code of the function itself in plain text such that it is executable using exec() in python. Any helper functions should be defined within the scope of the function 'evaluate_state'.
Include comments in your code so that it is readable, but everything should be implemented. The signature of the function should be as follows:

def evaluate_state(state) -> tuple[tuple[float, float], dict]:
    score_cards = state[0] # a python list of the score cards (integers) that have been played, in the order they were played, which may be of the same length as player_0_played_cards and player_1_played_cards or one more since the score card appears before the players play. May be empty
    player_0_played_cards = state[1] # a python list of the cards (integers) player 0 has played, in the order they were played. May be empty
    player_1_played_cards = state[2] # a python list of the cards (integers) player 1 has played, in the order they were played. Always same length as player_0_played_cards. May be empty
    is_turn = state[3] # bool, true if it is you and your opponent's turn to play, false if it is time to draw a new score card
    player_0_score = state[4] # float or integer, player 0's score so far
    player_1_score = state[5] #  float or integer, player 1's score so far
    score_deck = state[6] # a python set of the score cards (integers) left in the deck, either same length as player_0_hand and player_1_hand or one less since the score card appears before the players play. May be empty
    player_0_hand = state[7] # a python set of the cards (integers) left in player 0's hand. May be empty
    player_1_hand = state[8] # a python set of the cards (integers) left in player 1's hand. May be empty
    # explanation of what we do next
    ...
    <intermediate_value1> = value1 
    # explanation of what we do next
    ...
    <intermediate_value2> = value2 
    # explanation of what we do next
    ...
    player_scores = (player_0_expected_score, player_1_expected_score)
    intermediate_values = {'<intermediate_value1>': intermediate_value1, '<intermediate_value2>': intermediate_value2, ...}
    return player_scores, intermediate_values # make sure the return is exactly in this format

Where you can use your own names for the intermediate values and the values themselves.
Please start with "def evaluate_state(state):"
'''

# Do not include any other code, comments, or explanation in your output

HEURISTICS_FUNCTION_USAGE_PROMPTS = ['''Given the current situation, using the function defined, what is the value of the state?''']

REPRESENTATION_PROMPTS = ['''
What does this game state tell us about the current situation? Write down your thoughts and output the representation.
                          ''']

OPPONENT_ACTION_ENUMERATOR_PROMPT = '''Recall the rules of the game and enumerate the possible actions that the opponent could take currently. Write down your thoughts and output the list of actions.

Thought:
your thoughts here

Actions:
list of actions here, which should be a python list of card numbers, for example: [1, 2, 3]'''
OPPONENT_ACTION_PREDICTOR_PROMPT_bytes = b'''Given the current situation and what the opponent is trying to achieve, what is the probability of the opponent taking each action? Write down your thoughts and output the dict of probabilities

Thought:
your thoughts here

Probabilities:
gather your thoughts in a python dict, for example: {action1: prob1, action2: prob2, action3: prob3}'''

OPPONENT_ACTION_PREDICTOR_PROMPT = OPPONENT_ACTION_PREDICTOR_PROMPT_bytes.decode('utf-8')

ACTION_ENUMERATOR_PROMPT = '''Recall the rules of the game and enumerate the possible actions that we could take currently. Write down your thoughts and output the list of actions.

Thought:
your thoughts here

Actions:
list of actions here, which should be a python list of card numbers, for example: [1, 2, 3]'''


# %%
VERBALIZED_VALUE_PREDICOTR = """Here is a candidate state of the game, please analyze the chances of you winning th game with this state, then at the last line conclude \"Thus the value of the state for me is {{s}}\", where s is an integer from 1 to 10.
State:
The score cards have been played include {played_cards}. Score cards left in the deck include {score_cards}.
The cards you played include {your_cards}, thus cards left in your hand include {your_hand}.
The cards your opponent played include {opponent_cards}, thus cards left in your opponent's hand include {opponent_hand}.
In this state, the score you can get is {your_score}, and the score your opponent can get is {opponent_score}.

Analysis:
"""
# %%
VERBALIZED_OPACTION_PREDICTOR = """Here is a candidate state of the game, please analyze the chances of the opponent taking given actions with this state, then at the last line conclude \"Thus the probabilities of the actions of the opponent is {{s}}\", where s is a dictionary {{action1: prob1, ...}}.
State:
The score cards have been played include {played_cards}. Score cards left in the deck include {score_cards}.
The cards you played include {your_cards}, thus cards left in your hand include {your_hand}.
The cards your opponent played include {opponent_cards}, thus cards left in your opponent's hand include {opponent_hand}.

Possible Opponent Actions: {opponent_actions}

Analysis:
"""

STATE_PROMPT = """Here is a candidate state of the game.
State:
The score cards have been played include {played_cards}. Score cards left in the deck include {score_cards}.
The cards you played include {your_cards}, thus cards left in your hand include {your_hand}.
The cards your opponent played include {opponent_cards}, thus cards left in your opponent's hand include {opponent_hand}.
"""

# GOPS_RULES = """You are a player in a GOPS (Game of pure strategy) game. The game has two players, and is played with a deck of cards. Each player is dealt a hand of cards. \
# The goal of the game is to get the highest total scores. In each round, a player is asked to play a card from the hand to win the current score. The player who plays the highest card wins the round. \
# The player who wins the most scores wins the game.\
# Basically, you need to win the rounds with high scores. And you should also consider what cards left for you and your opponent to decide your strategy.\
# """

GOPS_RULES = """The game you want to write a function for is GOPS (game of pure strategy), also known as Goofspiel. The game has two players, and is played with a deck of score cards. Each player is dealt the same hand of cards at the beginning. The goal of the game is to get a score higher than your opponent. At the beginning of each round, a score card is randomly drawn without replacement from the score deck. Then each player plays a card simultaneously from their hand. The player who plays the higher card wins the round and gets the score card. They add the score of the score card to their total score. If the two cards played are the same, the person who wins the next round will get both score cards. The game continues until all score cards have been played. The player with the highest total score wins the game.\n"""

POLICY_FUNCTION_SIGNATURE = '''The function (written in python) should be named `policy_function` and take in a tuple called `state` of the game state as input. 
Specifically, the input tuple will be of length 9, with each element representing the following:
state[0]: a list of the score cards (integers) that have been played, in the order they were played
state[1]: a list of the cards (integers) player 0 has played, in the order they were played
state[2]: a list of the cards (integers) player 1 has played, in the order they were played
state[3]: boolean, true if it is you and your opponent's turn to play, false if it is time to draw a new score card
state[4]: float or integer, player 0's so far
state[5]: float or integer, player 1's score so far
state[6]: a set of the score cards (integers) left in the deck
state[7]: a set of the cards (integers) left in player 0's hand
state[8]: a set of the cards (integers) left in player 1's hand

It should return 2 elements. The first element is a dictionary mapping players to their optimal actions. For example, if you think player 0 should play card 1 and player 1 should play card 2, the function should return {{0: 1, 1: 2}}. The second element is a dictionary of intermediate values that you used to calculate the optimal actions. For example, if you used the value of the state to calculate the optimal actions, you should return {{'value': value}}.

Make sure your output only includes the code of the function itself in plain text such that it is executable using exec() in python. Any helper functions should be defined within the scope of the main function. 
Include comments in your code so that it is readable, but everything should be implemented. The signature of the function should be as follows:

def policy_function(state) -> tuple[dict[Any, Any], dict]:
    score_cards = state[0] # list
    player_0_played_cards = state[1] # list
    player_1_played_cards = state[2] # list
    is_turn = state[3] # bool
    player_0_score = state[4] # float or int
    player_1_score = state[5] # float or int
    score_deck = state[6] # set
    player_0_hand = state[7] # set
    player_1_hand = state[8] # set
    ...
    <intermediate_value1> = value1
    ...
    <intermediate_value2> = value2
    ...
    return {<player1>: <player1_optimal_action>, <player2>: <player2_optimal_action>}, {'<intermediate_value1>': intermediate_value1, '<intermediate_value2>': intermediate_value2, ...}

Where you can use your own names for the intermediate values and the values themselves.
Please start with "def policy_function(state):"
'''