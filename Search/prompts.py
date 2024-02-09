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

Points:
the number of points here''', 
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

Function:
the function here''']

GOPS_VALUE_FUNCTION_PROMPT = '''Convert the function you just described into python code.

The function should be named `evaluate_state` and take in a tuple called `state` of the game state as input. 
Specifically, the input tuple will be of length 6, with each element representing the following:
state[0]: the score cards that have been played, in the order they were played
state[1]: the cards you have played, in the order they were played
state[2]: the cards your opponent has played, in the order they were played
state[3]: true if it is you and your opponent's turn to play, false if it is time to draw a new score card
state[4]: your score so far
state[5]: your opponent's score so far

It should return a tuple of 2 elements, with the first element being the score you expect you will get at the end of the game, and the second element being the score you expect your opponent will get at the end of the game.
For example, if you think you will win 12 total points by the end of the game and your opponent will win 8 total points, the function should return (12, 8).

Make sure your output only includes the code of the function itself in plain text such that it is executable using exec() in python. 
Do not include any other code, comments, or explanation in your output.
'''

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

GOPS_RULES = """You are a player in a GOPS (Game of pure strategy) game. The game has two players, and is played with a deck of cards. Each player is dealt a hand of cards. \
The goal of the game is to get the highest total scores. In each round, a player is asked to play a card from the hand to win the current score. The player who plays the highest card wins the round. \
The player who wins the most scores wins the game.\
Basically, you need to win the rounds with high scores. And you should also consider what cards left for you and your opponent to decide your strategy.\
"""