def gen_seed_thought_prompt():
    # combine SYS_PROMPT, GOPS_RULES, HEURISTICS_SEED_PROMPT, and GOPS_FUNCTION_SIGNATURE
    return SYS_PROMPT + GOPS_RULES + HEURISTICS_SEED_THOUGHT_PROMPT 

def gen_seed_function_prompt(seed_thought_prompt):
    # combine seed_thought_prompt and function
    return seed_thought_prompt + "Implement the function you just described into python code." + GOPS_FUNCTION_SIGNATURE

def gen_feedback_analyze_prompt(function, feedback):
    '''
    tells the LLM to analyze the feedback
    '''
    # combine FEEDBACK_PROMPTS, function, feedback, and thoughts
    return SYS_PROMPT + GOPS_RULES + PREVIOUS_FUNCTION_INTRO + function + FEEDBACK_PROMPTS[1] + feedback + FEEDBACK_PROMPTS[5] + INPUT_BELOW_BREAKER

def gen_improvement_thought_prompt(previous, prev_thoughts):
    '''
    tells the LLM to improve upon previous thoughts
    '''
    # combine FEEDBACK_PROMPTS, function, feedback, and thoughts
    return previous + INPUT_ABOVE_BREAKER + FEEDBACK_PROMPTS[2] + prev_thoughts + INPUT_BELOW_BREAKER

# def gen_specific_improvement_prompt(function, feedback, num_improvements):
#     '''
#     tells the LLM to improve upon previous thoughts
#     '''
#     # combine FEEDBACK_PROMPTS, function, feedback, and thoughts
#     return SYS_PROMPT + GOPS_RULES + PREVIOUS_FUNCTION_INTRO + function + FEEDBACK_INTRO + feedback + f'''\n Based on the feedback given and the function you generated previously, what are {num_improvements} specific improvement we can make to the function to make it better? Be as specific and concrete as possible, and enumerate each improvement using 1.), 2.), ...\n''' # + INPUT_BELOW_BREAKER



def gen_specific_improvement_prompt(previous_prompt, conclusions):
    '''
    tells the LLM to improve upon previous thoughts
    '''
    # combine FEEDBACK_PROMPTS, function, feedback, and thoughts
    return previous_prompt + conclusions + SPECIFIC_IMPROVEMENT_FROM_CONCLUSION

def gen_draw_conclusions_from_feedback_prompt(function, feedback):
    '''
    tells the LLM to draw conclusions from feedback
    '''
    # combine FEEDBACK_PROMPTS, function, feedback, and thoughts
    return SYS_PROMPT + GOPS_RULES + PREVIOUS_FUNCTION_INTRO + function + FEEDBACK_INTRO + feedback + CONCLUSION_FROM_FEEDBACK #+ INPUT_BELOW_BREAKER

def gen_implement_function_from_improvement_prompt(function, idea):
    '''
    tells the LLM to implement the function from the improvement
    '''
    # combine FEEDBACK_PROMPTS, function, feedback, and thoughts
    return SYS_PROMPT + GOPS_RULES + PREVIOUS_FUNCTION_INTRO + function + "Here is a possible way to improve this function: \n" + idea + "\n Implement this improvement into the function as best as you can. Make sure not to change the function signature, which we reproduce below: \n" + GOPS_FUNCTION_SIGNATURE

# def gen_new_improvements_prompt(thought_prompt, new_thoughts):
#     # combine thought_prompt and new_thoughts
#     return INPUT_ABOVE_BREAKER + thought_prompt + new_thoughts + FEEDBACK_PROMPTS[3] + INPUT_BELOW_BREAKER

def gen_improved_function_prompt(previous):
    '''
    tells the LLM to generate improved function
    '''
    # combine new_improvements_prompt and new_improvements
    return previous + INPUT_ABOVE_BREAKER + FEEDBACK_PROMPTS[4] + GOPS_FUNCTION_SIGNATURE + INPUT_BELOW_BREAKER

def gen_execution_error_feedback(previous, function, execution_error):
    return previous + INPUT_ABOVE_BREAKER + f"You previously generated the following improved function based on your thoughts and analysis of the feedback: \n {function} \n However, it ran into the following error when running: {execution_error} \n Please fix the error and output your function again. Recall that: " + GOPS_FUNCTION_SIGNATURE + INPUT_BELOW_BREAKER

def gen_execution_error_feedback_2(error_message):
    string = f'''There was an execution error when running the function you generated on test states. 
    The error message was:
    {error_message}
    Please fix the error and try again.
    '''
    return string


def gen_single_state_example_feedback(i, state_description, estimated_score, intermediate_values, search_score, actual_score): 
    string = f'''--------------------------
    Example {i}:
    The state you were trying to estimate a value for is:
    {state_description}

    The function you generated returned the following values:
    {estimated_score}
    for the expected end of game scores of the players. 

    Some intermediate values that you used to calculate the scores were:
    {intermediate_values}

    The estimated end of game scores of the players using lookahead search with your function was:
    {search_score}

    The actual scores of the players at the end of the game in the simulation were:
    {actual_score}
    --------------------------
    '''
    return string

INPUT_ABOVE_BREAKER = '''\n <<<<<<<<<<<<<<<<<<<<<<<<<< \n'''

INPUT_BELOW_BREAKER = '''\n >>>>>>>>>>>>>>>>>>>>>>>>>> \n'''

BLOCK_BREAKER = '''\n ------------------------ \n'''

SYS_PROMPT = '''You are a function engineer trying to write a function that can evaluate the value of a state in a game. This is known as a value heuristic, and will be used in lookahead search algorithms to evaluate the value of unexplored states. Your goal is to develop a heuristic that is as accurate as possible without being too expensive to compute. Hence, you are not allowed to runs simulations in the function.\n'''

SYS_PROMPT_POLICY = '''You are a function engineer trying to write a function that outputs the optimal action for each player given a state in a game. This is known as a policy function, and will be used by an agent to play the game. Your goal is to develop a policy function that that produces the best possible actions without being too expensive to compute. Hence, you are not allowed to runs simulations in the function.\n'''

GOPS_RULES = '''The game you want to write a function for is GOPS (game of pure strategy), also known as Goofspiel. The game has two players, and is played with a deck of score cards. Each player is dealt the same hand of cards at the beginning. The goal of the game is to get a score higher than your opponent. At the beginning of each round, a score card is randomly drawn without replacement from the score deck. Then each player plays a card simultaneously from their hand. The player who plays the higher card wins the round and gets the score card. They add the score of the score card to their total score. If the two cards played are the same, the person who wins the next round will get both score cards. The game continues until all score cards have been played. The player with the highest total score wins the game.\n'''



HEURISTICS_SEED_THOUGHT_PROMPT =  '''Given the rules of the game, come up with a function that can be used to evaluate the value of a state in the game.
Write down your thoughts and pseudocode for the function.

-----
Thoughts:
<your thoughts here. try to think about characteristics of the game that can help you>
-----
Pseudocode: 
<the pseudocode for your function here. You can be as abstract as you want>
-----

Below is an example for the game of tic-tac-toe, where I calculate the probability of winning for each player.
                               
"
-----
Thoughts:
- If I have 3 in a row, I win
- If my opponent has 3 in a row, I lose
- Otherwise, the game is a draw
- The center position is the most important position         
-----                                                     
Pseudocode:            
Use an if-else function to check the following,
- If it is my turn check if I can win, if so return (1.0,0.0)
- If it is my opponent's turn check if they can win, if so return (0.0,1.0)
- If I control the center or it is my turn and the center is empty, return (0.8,0.2)
- If the opponent controls the center or it is my opponent's turn and the center is empty, return (0.2,0.8)
-----                                 
"
                 
Remember that the function should output the score that you expect to get at the end of the game and the score that you expect your opponent will get at the end of the game. 
For example, if you think you will win 12 total points by the end of the game and your opponent will win 8 total points, the function should return (12, 8).'''

HEURISTICS_SEED_THOUGHT_PROMPT_2 =  '''Given the rules of the game, come up with a function that can be used to evaluate the value of a state in the game.
Write down your thoughts and pseudocode for the function.

-----
Thoughts:
<your thoughts here. try to think about characteristics of the game that can help you>
-----
Pseudocode: 
<the pseudocode for your function here. You can be as abstract as you want>
-----

Below is an example for the game of GOPS (Goofspiel), where I calculate the expected score for each player at the end of the game.
                               
"
Thoughts:
In GOPS, the value of a state can be determined by the current total score of each player, the remaining score cards in the deck, and the cards left in each player's hand.\n- Winning a round with a high score card can significantly impact the total score, so having high-value cards left in hand is important.\n- The distribution of score cards in the deck can also affect the value of a state, as certain cards may be more valuable than others.\n\nPseudocode:\n1. Define a function evaluate_state(state) that takes the current state of the game as input.\n2. Calculate the total score of each player based on the current state.\n3. Determine the remaining score cards in the deck and the cards left in each player's hand.\n4. Evaluate the potential value of the state by considering factors such as:\n   - The difference in total scores between the players\n   - The value of the cards left in each player's hand\n   - The distribution of score cards in the deck\n5. Return a tuple containing the expected score for the current player and the opponent player at the end of the game.\n
Pseudocode:
```\nfunction evaluate_state(state):\n    player_score = calculate_total_score(state, player)\n    opponent_score = calculate_total_score(state, opponent)\n    \n    remaining_score_cards = get_remaining_score_cards(state)\n    player_hand = get_player_hand(state, player)\n    opponent_hand = get_player_hand(state, opponent)\n    \n    player_potential_score = calculate_potential_score(player_score, player_hand, remaining_score_cards)\n    opponent_potential_score = calculate_potential_score(opponent_score, opponent_hand, remaining_score_cards)\n    \n    return (player_potential_score, opponent_potential_score)                          
"
                 
Remember that the function should output the expected score for all players at the end of the game, one for each player.'''

HEURISTICS_SEED_THOUGHT_PROMPTS = [HEURISTICS_SEED_THOUGHT_PROMPT, HEURISTICS_SEED_THOUGHT_PROMPT_2]

IMPROVEMENT_PROMPTS = [
    "Write a new Python function that improves upon the function given below. Make sure to include comments in your code so that it is readable describing what you improved on a why. Your output should ONLY include the new function, with any comments written as comments in the code. Also, do not change the name of the function.",
    "Below is a reward function for the game of GOPS (game of pure strategy) that is used in reinforcement learning and Monte Carlo Tree Search to evaluate the value of a state. Write a new Python function that improves upon the function given below. Make sure to include comments in your code so that it is readable describing what you improved on a why. Your output should ONLY include the new function, with any comments written as comments in the code. Also, do not change the name of the function.",
]

PREVIOUS_FUNCTION_INTRO = '''Previously you generated the following function to evaluate the value of a state in the game of GOPS.

    Previous function: \n'''

FEEDBACK_INTRO = '''\n Below is some feedback on how the function you generated performed when we tested it. Note that simulations involve high variance and the actual scores may not match the expected scores exactly.
    
    Feedback: \n'''

FEEDBACK_INTRO = '''Below is some feedback on how the function you generated performed when we tested it. Note that simulations involve high variance and the actual scores may not match the expected scores exactly.
    
    Feedback: \n'''

MODIFY_ABSTRACT = '''Given the conclusions you drew from the feedback, modify your previous thoughts and pseudocode accordingly. Notate very clearly what parts of your previous thoughts and pseudocode that you modified and why. We reproduce your previous thoughts and pseudocode below for your reference. \n'''

# Based on the feedback given and your thoughts, also make a list of the possible areas of the function that you can improve. For each area, write a brief description of how you can improve it and why this would help the function achieve its purpose better.

MODIFY_FUNCTION = '''Now modify the function you generated previously to improve it based on the feedback given and your thoughts, and output the new function.\n'''

# FEEDBACK_PROMPTS = [
#     '''
#     Previously you generated the following function to evaluate the value of a state in the game of GOPS.

#     Previous function: \n
#     ''',
#     '''\n Below is some feedback on how the function you generated performed when we tested it. Note that simulations involve high variance and the actual scores may not match the expected scores exactly.
    
#     Feedback: \n''',
#     '''Given the conclusions you drew from the feedback, modify your previous thoughts and pseudocode accordingly. Based on the feedback given and your thoughts, also make a list of the possible areas of the function that you can improve. For each area, write a brief description of how you can improve it and why this would help the function achieve its purpose better. We reproduce your previous thoughts and pseudocode below for your reference. \n''',
#     '''Based on the feedback given and your thoughts, make a list of the possible areas of the function that you can improve. For each area, write a brief description of how you can improve it and why this would help the function achieve its purpose better. \n''',
#     '''Now modify the function you generated previously to improve it based on the feedback given and your thoughts, and output the new function.\n''',
#     '''Please think about what conclusions we can draw from this feedback. What parts of the function do you think we can adjust so that it performs better? Be concrete and specific, citing the feedback examples given.\n'''
# ]

CONCLUSION_FROM_FEEDBACK = '''Based on the feedback given and the function you generated previously, what are some conclusions you can draw from the feedback? Make sure to cite the specific examples in the feedback to justify your analysis.\n '''

SPECIFIC_IMPROVEMENT_FROM_CONCLUSION = '''Based on the function, feedback, and conclusions you drew, what is one improvement that you can make to the function that you think will have the most impact? Be as specific and concrete as possible.\n '''

GOPS_FUNCTION_SIGNATURE = '''The function (written in python) should be named `evaluate_state` and take in a tuple called `state` of the game state as input. 
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

It should return 2 elements. 
The first element should be a tuple with 2 floats: the first element being the score you expect player 0 will get at the end of the game, and the second element being the score you expect player 1 will get at the end of the game.
The second element should be a dictionary of any important intermediate values that you used to calculate the scores.
For example, if you think player 0 will win 12 total points by the end of the game and player 1 will win 8 total points, the function should return (12, 8).

Make sure your output only includes the code of the function itself in plain text such that it is executable using exec() in python. Any helper functions should be defined within the scope of the function 'evaluate_state'.
Include comments in your code so that it is readable, but everything should be implemented. The signature of the function should be as follows:

def evaluate_state(state) -> tuple[tuple[float, float], dict]:
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
    return (player_0_expected_score, player_1_expected_score), {'<intermediate_value1>': intermediate_value1, '<intermediate_value2>': intermediate_value2, ...}

Where you can use your own names for the intermediate values and the values themselves.
Please start with "def evaluate_state(state):"
'''
