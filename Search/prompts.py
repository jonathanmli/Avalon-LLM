HIDDEN_STATE_ENUMERATOR_PROMPT = '''Recall the rules of the game and enumerate the possible hidden states of the game that we could be in.'''
HIDDEN_STATE_PREDICTOR_PROMPT = '''Given the current situation, for each of the hidden states, what is the probability of being in that state?'''
HIDDEN_STATE_PREDICTOR_PROMPT_SINGLE = '''Given the current situation, on a scale of 0.0 to 1.0, what is the likelihood of being this hidden state?'''

FORWARD_ENUMERATOR_PROMPT = '''Recall the rules of the game and enumerate the possible next states of the game that we could be in given the current state and action taken.'''
FORWARD_PREDICTOR_PROMPT = '''Given the current state and action taken, for each of the next states, what is the probability of being in that state?'''

VALUE_PREDICTOR_PROMPTS = ['''Given the current situation, what is the probability of winning?''', '''Given the current situation, what is the value of the game?''']

OPPONENT_ACTION_ENUMERATOR_PROMPT = '''Recall the rules of the game and enumerate the possible actions that the opponent could take currently.'''
OPPONENT_ACTION_PREDICTOR_PROMPT = '''Given the current situation and what the opponent is trying to achieve, what is the probability of the opponent taking each action?'''

ACTION_ENUMERATOR_PROMPT = '''Recall the rules of the game and enumerate the possible actions that we could take currently.'''