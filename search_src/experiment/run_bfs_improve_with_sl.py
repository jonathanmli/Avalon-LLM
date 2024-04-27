from src.self_improve.search_improve import bfs_improve
from src.GOPS.examples.func_list import *

TEST_FUNCTION = """
def evaluate_state(state):
    # Extracting the relevant information from the state tuple
    score_cards = state[0]
    my_cards = state[1]
    opponent_cards = state[2]
    is_turn = state[3]
    my_score = state[4]
    opponent_score = state[5]
    deck = state[6]
    
    # Calculating the chances of winning a round based on the cards in hand
    my_chances = 0
    opponent_chances = 0
    
    for card in my_cards:
        if card > max(opponent_cards):
            my_chances += 1
    
    for card in opponent_cards:
        if card > max(my_cards):
            opponent_chances += 1
    
    # Updating the scores based on the chances of winning
    my_score += my_chances
    opponent_score += opponent_chances
    
    # Returning the updated scores
    return (my_score, opponent_score)"""

def main():
    seed_functions = [test_func]
    graph = bfs_improve(seed_functions)
    print(graph)
    print(graph.get_highest_value_state(0))

if __name__ == '__main__':
    main()