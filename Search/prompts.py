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
the probability here''', '''Given the current situation, what is the value of the game? Write down your thoughts and rate on a scale from 0 (no value) to 10 (extremely valuable).

Thoughts:
your thoughts here

Value:
the value here''']

OPPONENT_ACTION_ENUMERATOR_PROMPT = '''Recall the rules of the game and enumerate the possible actions that the opponent could take currently. Write down your thoughts and output the list of actions.

Thought:
your thoughts here

Actions:
list of actions here, which should be a python list of card numbers, for example: [1, 2, 3]'''
OPPONENT_ACTION_PREDICTOR_PROMPT = '''Given the current situation and what the opponent is trying to achieve, what is the probability of the opponent taking each action?'''

ACTION_ENUMERATOR_PROMPT = '''Recall the rules of the game and enumerate the possible actions that we could take currently. Write down your thoughts and output the list of actions.

Thought:
your thoughts here

Actions:
list of actions here, which should be a python list of card numbers, for example: [1, 2, 3]'''