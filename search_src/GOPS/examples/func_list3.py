func_list = [
    """def evaluate_state(state) -> tuple[tuple[float, float], dict]:
    score_cards = state[0]
    player_0_played_cards = state[1]
    player_1_played_cards = state[2]
    is_turn = state[3]
    player_0_score = state[4]
    player_1_score = state[5]
    score_deck = state[6]
    player_0_hand = state[7]
    player_1_hand = state[8]
    
    # Calculate the total score of each player
    def calculate_total_score(state, player):
        if player == 0:
            return state[4]
        else:
            return state[5]
    
    # Get the remaining score cards in the deck
    remaining_score_cards = set(score_deck)
    
    # Calculate the potential score for a player based on their current state
    def calculate_potential_score(player_score, player_hand, remaining_score_cards):
        potential_score = player_score
        for card in player_hand:
            if card in remaining_score_cards:
                potential_score += card
        return potential_score
    
    # Evaluate the potential value of the state
    player_potential_score = calculate_potential_score(player_0_score, player_0_hand, remaining_score_cards)
    opponent_potential_score = calculate_potential_score(player_1_score, player_1_hand, remaining_score_cards)
    
    return (player_potential_score, opponent_potential_score), {'player_potential_score': player_potential_score, 'opponent_potential_score': opponent_potential_score}""",
    """def evaluate_state(state) -> tuple[tuple[float, float], dict]:
    score_cards = state[0]
    player_0_played_cards = state[1]
    player_1_played_cards = state[2]
    is_turn = state[3]
    player_0_score = state[4]
    player_1_score = state[5]
    score_deck = state[6]
    player_0_hand = state[7]
    player_1_hand = state[8]
    
    # Calculate the potential scores for each player based on the remaining score cards
    player_potential_score = player_0_score
    opponent_potential_score = player_1_score
    
    if len(score_deck) > 0:
        remaining_score_cards = len(score_deck)
        if len(player_0_played_cards) > len(player_1_played_cards):
            player_potential_score += remaining_score_cards / 2
        elif len(player_1_played_cards) > len(player_0_played_cards):
            opponent_potential_score += remaining_score_cards / 2
    
    player_advantage = player_potential_score - opponent_potential_score
    
    return (player_potential_score, opponent_potential_score), {'player_advantage': player_advantage}""",
    """def evaluate_state(state) -> tuple[tuple[float, float], dict]:
    # Extracting necessary information from the state tuple
    score_cards = state[0]
    player_0_played_cards = state[1]
    player_1_played_cards = state[2]
    is_turn = state[3]
    player_0_score = state[4]
    player_1_score = state[5]
    score_deck = state[6]
    player_0_hand = state[7]
    player_1_hand = state[8]
    
    # Function to calculate total score of a player
    def calculate_total_score(player_played_cards):
        return sum(player_played_cards)
    
    # Calculate total scores of each player
    player_0_total_score = calculate_total_score(player_0_played_cards)
    player_1_total_score = calculate_total_score(player_1_played_cards)
    
    # Calculate the difference in total scores between the players
    score_difference = player_0_total_score - player_1_total_score
    
    # Count the remaining score cards in the deck
    remaining_score_cards = len(score_deck)
    
    # Calculate the potential score for each player based on their remaining hand
    player_0_potential_score = sum(player_0_hand)
    player_1_potential_score = sum(player_1_hand)
    
    # Check if there are consecutive rounds won by a player
    def check_consecutive_rounds(player_played_cards):
        consecutive_rounds = 0
        for i in range(len(player_played_cards) - 1):
            if player_played_cards[i] > player_played_cards[i + 1]:
                consecutive_rounds += 1
        return consecutive_rounds
    
    player_0_consecutive_rounds = check_consecutive_rounds(player_0_played_cards)
    player_1_consecutive_rounds = check_consecutive_rounds(player_1_played_cards)
    
    # Calculate the value of the state based on the factors
    state_value = score_difference + remaining_score_cards + player_0_potential_score - player_1_potential_score + player_0_consecutive_rounds - player_1_consecutive_rounds
    
    # Return the expected scores for each player at the end of the game
    return (player_0_score + state_value, player_1_score - state_value), {'score_difference': score_difference, 'remaining_score_cards': remaining_score_cards, 'player_0_potential_score': player_0_potential_score, 'player_1_potential_score': player_1_potential_score, 'player_0_consecutive_rounds': player_0_consecutive_rounds, 'player_1_consecutive_rounds': player_1_consecutive_rounds}""",
    """def evaluate_state(state) -> tuple[tuple[float, float], dict]:
    score_cards = state[0]
    player_0_played_cards = state[1]
    player_1_played_cards = state[2]
    is_turn = state[3]
    player_0_score = state[4]
    player_1_score = state[5]
    score_deck = state[6]
    player_0_hand = state[7]
    player_1_hand = state[8]
    
    # Helper function to calculate the total score of a player
    def calculate_player_score(player_played_cards):
        return sum(player_played_cards)
    
    # Calculate the total scores of both players
    player_0_total_score = calculate_player_score(player_0_played_cards) + player_0_score
    player_1_total_score = calculate_player_score(player_1_played_cards) + player_1_score
    
    # Calculate the difference in scores between the two players
    score_difference = player_0_total_score - player_1_total_score
    
    # Calculate the number of score cards remaining in the deck
    remaining_score_cards = len(score_deck)
    
    # Intermediate values for analysis
    intermediate_values = {
        'player_0_total_score': player_0_total_score,
        'player_1_total_score': player_1_total_score,
        'score_difference': score_difference,
        'remaining_score_cards': remaining_score_cards
    }
    
    # Estimate the expected total score for each player at the end of the game
    player_0_expected_score = player_0_total_score + (score_difference / 2) + (remaining_score_cards / 2)
    player_1_expected_score = player_1_total_score - (score_difference / 2) - (remaining_score_cards / 2)
    
    return (player_0_expected_score, player_1_expected_score), intermediate_values""",
    """def evaluate_state(state) -> tuple[tuple[float, float], dict]:
    score_cards = state[0]
    player_0_played_cards = state[1]
    player_1_played_cards = state[2]
    is_turn = state[3]
    player_0_score = state[4]
    player_1_score = state[5]
    score_deck = state[6]
    player_0_hand = state[7]
    player_1_hand = state[8]
    
    player_0_expected_score = player_0_score
    player_1_expected_score = player_1_score
    
    high_value_cards = {card for card in score_deck if card > 10}
    
    for card in score_deck:
        if card in high_value_cards:
            player_0_expected_score += 0.6
            player_1_expected_score += 0.4
        else:
            player_0_expected_score += 0.4
            player_1_expected_score += 0.6
    
    intermediate_values = {'high_value_cards': high_value_cards}
    
    return (player_0_expected_score, player_1_expected_score), intermediate_values"""
]

func_0412 = [
    """def evaluate_state(state) -> tuple[tuple[float, float], dict]:
    # Get the state variables
    score_cards = state[0]
    player_0_played_cards = state[1]
    player_1_played_cards = state[2]
    is_turn = state[3]
    player_0_score = state[4]
    player_1_score = state[5]
    score_deck = state[6]
    player_0_hand = state[7]
    player_1_hand = state[8]

    # Helper function to check if the game has ended
    def game_has_ended(score_deck, player_0_hand, player_1_hand):
        return len(score_deck) == 0 and len(player_0_hand) == 0 and len(player_1_hand) == 0

    # Helper function to calculate the potential score for a player
    def calculate_potential_score(player_score, player_hand, remaining_score_cards):
        potential_score = player_score
        for card in player_hand:
            if card in remaining_score_cards:
                potential_score += card
        return potential_score

    # Helper function to calculate the score potential of the cards in a player's hand
    def calculate_hand_potential(player_hand, remaining_score_cards):
        hand_potential = 0
        for card in player_hand:
            if card in remaining_score_cards:
                hand_potential += card
        return hand_potential

    # If the game has already ended, return the final scores of the players
    if game_has_ended(score_deck, player_0_hand, player_1_hand):
        return (player_0_score, player_1_score), {}

    # Calculate the potential score for each player
    player_potential_score = calculate_potential_score(player_0_score, player_0_hand, score_deck)
    opponent_potential_score = calculate_potential_score(player_1_score, player_1_hand, score_deck)

    # Calculate the potential of the cards in each player's hand
    player_hand_potential = calculate_hand_potential(player_0_hand, score_deck)
    opponent_hand_potential = calculate_hand_potential(player_1_hand, score_deck)

    # Consider the current score difference between the players
    score_difference = player_0_score - player_1_score

    # Consider the potential of the cards in each player's hand
    hand_potential_difference = player_hand_potential - opponent_hand_potential

    # Adjust the potential score for each player based on the score difference and the hand potential difference
    # Use a scaling factor to adjust the estimated scores based on the current score difference
    scaling_factor = 0.5
    player_potential_score += scaling_factor * (score_difference + hand_potential_difference)
    opponent_potential_score -= scaling_factor * (score_difference + hand_potential_difference)

    return (player_potential_score, opponent_potential_score), {'player_potential_score': player_potential_score, 'opponent_potential_score': opponent_potential_score, 'score_difference': score_difference, 'hand_potential_difference': hand_potential_difference}""",
    """def evaluate_state(state) -> tuple[tuple[float, float], dict]:
    score_cards = state[0] # list
    player_0_played_cards = state[1] # list
    player_1_played_cards = state[2] # list
    is_turn = state[3] # bool
    player_0_score = state[4] # float or int
    player_1_score = state[5] # float or int
    score_deck = state[6] # set
    player_0_hand = state[7] # set
    player_1_hand = state[8] # set

    # Helper Function to calculate the potential score
    def calculate_potential_score(player_score, player_hand, remaining_score_cards, opponent_hand):
        potential_score = player_score
        for card in player_hand:
            if card in remaining_score_cards and card > min(opponent_hand):
                potential_score += card
        return potential_score

    # Helper Function to calculate the value of hand
    def calculate_hand_value(player_hand):
        return sum(player_hand)

    # Helper Function to calculate the turn bonus
    def calculate_turn_bonus(remaining_score_cards, player_hand):
        if len(remaining_score_cards) > 0 and len(player_hand) > 0:
            return (sum(remaining_score_cards) / len(remaining_score_cards)) + (sum(player_hand) / len(player_hand))
        else:
            return 0

    # Calculate the potential score and hand value of each player
    player_potential_score = calculate_potential_score(player_0_score, player_0_hand, score_deck, player_1_hand)
    opponent_potential_score = calculate_potential_score(player_1_score, player_1_hand, score_deck, player_0_hand)
    player_hand_value = calculate_hand_value(player_0_hand)
    opponent_hand_value = calculate_hand_value(player_1_hand)

    # Adjust the potential score based on the current turn
    turn_bonus = calculate_turn_bonus(score_deck, player_0_hand if is_turn else player_1_hand)
    player_potential_score += turn_bonus if is_turn else 0
    opponent_potential_score += turn_bonus if not is_turn else 0

    # Include the hand value in the final expected score
    player_expected_score = player_potential_score + player_hand_value
    opponent_expected_score = opponent_potential_score + opponent_hand_value

    # Return the expected score along with the intermediate values used in the calculation
    return (player_expected_score, opponent_expected_score), {'player_potential_score': player_potential_score, 'opponent_potential_score': opponent_potential_score, 'player_hand_value': player_hand_value, 'opponent_hand_value': opponent_hand_value}""",
    """def evaluate_state(state) -> tuple[tuple[float, float], dict]:
    score_cards = state[0] # list
    player_0_played_cards = state[1] # list
    player_1_played_cards = state[2] # list
    is_turn = state[3] # bool
    player_0_score = state[4] # float or int
    player_1_score = state[5] # float or int
    score_deck = state[6] # set
    player_0_hand = state[7] # set
    player_1_hand = state[8] # set

    # Helper Function to calculate the strategic value of a card
    def calculate_card_value(card, opponent_hand, remaining_score_cards):
        card_value = 0
        for opponent_card in opponent_hand:
            if card > opponent_card and card in remaining_score_cards:
                card_value += card
        return card_value

    # Helper Function to calculate the potential score
    def calculate_potential_score(player_score, player_hand, remaining_score_cards, opponent_hand):
        potential_score = player_score
        for card in player_hand:
            potential_score += calculate_card_value(card, opponent_hand, remaining_score_cards)
        return potential_score

    # Helper Function to calculate the turn bonus
    def calculate_turn_bonus(remaining_score_cards, player_hand):
        if len(remaining_score_cards) > 0 and len(player_hand) > 0:
            return (sum(remaining_score_cards) / len(remaining_score_cards)) + (sum(player_hand) / len(player_hand))
        else:
            return 0

    # Helper Function to calculate the strategic value of hand
    def calculate_hand_strategic_value(player_hand, opponent_hand, remaining_score_cards):
        hand_value = 0
        for card in player_hand:
            hand_value += calculate_card_value(card, opponent_hand, remaining_score_cards)
        return hand_value

    # Calculate the potential score, hand value and strategic value of each player
    player_potential_score = calculate_potential_score(player_0_score, player_0_hand, score_deck, player_1_hand)
    opponent_potential_score = calculate_potential_score(player_1_score, player_1_hand, score_deck, player_0_hand)
    player_strategic_value = calculate_hand_strategic_value(player_0_hand, player_1_hand, score_deck)
    opponent_strategic_value = calculate_hand_strategic_value(player_1_hand, player_0_hand, score_deck)

    # Adjust the potential score based on the current turn
    turn_bonus = calculate_turn_bonus(score_deck, player_0_hand if is_turn else player_1_hand)
    player_potential_score += turn_bonus if is_turn else 0
    opponent_potential_score += turn_bonus if not is_turn else 0

    # Include the hand value in the final expected score
    player_expected_score = player_potential_score + player_strategic_value
    opponent_expected_score = opponent_potential_score + opponent_strategic_value

    # Return the expected score along with the intermediate values used in the calculation
    return (player_expected_score, opponent_expected_score), {'player_potential_score': player_potential_score, 'opponent_potential_score': opponent_potential_score, 'player_strategic_value': player_strategic_value, 'opponent_strategic_value': opponent_strategic_value}""",
    # gpt-3.5 old
    """def evaluate_state(state) -> tuple[tuple[float, float], dict]:
    score_cards = state[0]
    player_0_played_cards = state[1]
    player_1_played_cards = state[2]
    is_turn = state[3]
    player_0_score = state[4]
    player_1_score = state[5]
    score_deck = state[6]
    player_0_hand = state[7]
    player_1_hand = state[8]
    
    # Calculate the total score of each player
    def calculate_total_score(state, player):
        if player == 0:
            return state[4]
        else:
            return state[5]
    
    # Get the remaining score cards in the deck
    remaining_score_cards = set(score_deck)
    
    # Calculate the potential score for a player based on their current state
    def calculate_potential_score(player_score, player_hand, remaining_score_cards):
        potential_score = player_score
        for card in player_hand:
            if card in remaining_score_cards:
                potential_score += card
        return potential_score
    
    # Evaluate the potential value of the state
    player_potential_score = calculate_potential_score(player_0_score, player_0_hand, remaining_score_cards)
    opponent_potential_score = calculate_potential_score(player_1_score, player_1_hand, remaining_score_cards)
    
    return (player_potential_score, opponent_potential_score), {'player_potential_score': player_potential_score, 'opponent_potential_score': opponent_potential_score}""",
    """def evaluate_state(state) -> tuple[tuple[float, float], dict]:
    # Extracting necessary information from the state tuple
    score_cards = state[0]
    player_0_played_cards = state[1]
    player_1_played_cards = state[2]
    is_turn = state[3]
    player_0_score = state[4]
    player_1_score = state[5]
    score_deck = state[6]
    player_0_hand = state[7]
    player_1_hand = state[8]
    
    # Function to calculate total score of a player
    def calculate_total_score(player_played_cards):
        return sum(player_played_cards)
    
    # Calculate total scores of each player
    player_0_total_score = calculate_total_score(player_0_played_cards)
    player_1_total_score = calculate_total_score(player_1_played_cards)
    
    # Calculate the difference in total scores between the players
    score_difference = player_0_total_score - player_1_total_score
    
    # Count the remaining score cards in the deck
    remaining_score_cards = len(score_deck)
    
    # Calculate the potential score for each player based on their remaining hand
    player_0_potential_score = sum(player_0_hand)
    player_1_potential_score = sum(player_1_hand)
    
    # Check if there are consecutive rounds won by a player
    def check_consecutive_rounds(player_played_cards):
        consecutive_rounds = 0
        for i in range(len(player_played_cards) - 1):
            if player_played_cards[i] > player_played_cards[i + 1]:
                consecutive_rounds += 1
        return consecutive_rounds
    
    player_0_consecutive_rounds = check_consecutive_rounds(player_0_played_cards)
    player_1_consecutive_rounds = check_consecutive_rounds(player_1_played_cards)
    
    # Calculate the value of the state based on the factors
    state_value = score_difference + remaining_score_cards + player_0_potential_score - player_1_potential_score + player_0_consecutive_rounds - player_1_consecutive_rounds
    
    # Return the expected scores for each player at the end of the game
    return (player_0_score + state_value, player_1_score - state_value), {'score_difference': score_difference, 'remaining_score_cards': remaining_score_cards, 'player_0_potential_score': player_0_potential_score, 'player_1_potential_score': player_1_potential_score, 'player_0_consecutive_rounds': player_0_consecutive_rounds, 'player_1_consecutive_rounds': player_1_consecutive_rounds}""",
    """def evaluate_state(state) -> tuple[tuple[float, float], dict]:
    score_cards = state[0] 
    player_0_played_cards = state[1] 
    player_1_played_cards = state[2] 
    is_turn = state[3] 
    player_0_score = state[4] 
    player_1_score = state[5] 
    score_deck = state[6] 
    player_0_hand = state[7] 
    player_1_hand = state[8] 
    
    # Calculate the total score of each player
    def calculate_total_score(state, player):
        if player == 0:
            return state[4]
        else:
            return state[5]
    
    # Get the remaining score cards in the deck
    remaining_score_cards = set(score_deck)
    
    # Calculate the potential score for a player based on their current state
    def calculate_potential_score(player_score, player_hand, remaining_score_cards):
        potential_score = player_score
        for card in player_hand:
            if card in remaining_score_cards:
                potential_score += card
        return potential_score
    
    # Evaluate the potential value of the state with improvements
    player_potential_score = calculate_potential_score(player_0_score, player_0_hand, remaining_score_cards)
    opponent_potential_score = calculate_potential_score(player_1_score, player_1_hand, remaining_score_cards)
    
    # Include more sophisticated algorithms for improved evaluation
    # Include advanced lookahead search algorithm for better predictions
    
    # Adjusted calculations based on feedback
    player_expected_score = player_potential_score + 2  # Adjusted calculation for player's expected score
    opponent_expected_score = opponent_potential_score - 2  # Adjusted calculation for opponent's expected score
    
    # Intermediate values for transparency
    intermediate_values = {'player_potential_score': player_potential_score, 'opponent_potential_score': opponent_potential_score}
    
    return (player_expected_score, opponent_expected_score), intermediate_values
    """,
    # gpt-3.5 new
    # """def evaluate_state(state) -> tuple[tuple[float, float], dict]:
    # score_cards = state[0] # List of score cards that have been played
    # player_0_played_cards = state[1] # List of cards player 0 has played
    # player_1_played_cards = state[2] # List of cards player 1 has played
    # is_turn = state[3] # Boolean indicating turn
    # player_0_score = state[4] # Player 0's score so far
    # player_1_score = state[5] # Player 1's score so far
    # score_deck = state[6] # Set of score cards left in the deck
    # player_0_hand = state[7] # Set of cards left in player 0's hand
    # player_1_hand = state[8] # Set of cards left in player 1's hand

    # def calculate_win_probabilities(player_0_cards, player_1_cards):
    #     # Function to calculate the probability of player 0 winning a round based on remaining cards
        
    #     total_rounds = len(score_cards) + 1
    #     player_0_wins = 0
    #     player_1_wins = 0
        
    #     # Simulate all possible outcomes based on remaining cards
    #     for card_0 in player_0_cards:
    #         for card_1 in player_1_cards:
    #             if card_0 > card_1:
    #                 player_0_wins += 1
    #             elif card_0 < card_1:
    #                 player_1_wins += 1
    #             # If cards are equal, simulate the next round
    #             else:
    #                 if total_rounds % 2 == 0: # Player 0's turn to play
    #                     player_0_wins += 0.5
    #                     player_1_wins += 0.5
    #                 else: # Player 1's turn to play
    #                     player_0_wins += 0.5
    #                     player_1_wins += 0.5
        
    #     # Avoid division by zero by checking if there are any possible outcomes
    #     if player_0_wins + player_1_wins == 0:
    #         return 0.5, 0.5
        
    #     # Calculate win probabilities
    #     player_0_win_prob = player_0_wins / (player_0_wins + player_1_wins)
    #     player_1_win_prob = player_1_wins / (player_0_wins + player_1_wins)
        
    #     return player_0_win_prob, player_1_win_prob

    # # Calculate win probabilities based on remaining cards in each player's hand
    # player_0_win_prob, player_1_win_prob = calculate_win_probabilities(player_0_hand, player_1_hand)

    # # Estimate final scores based on win probabilities
    # player_0_expected_score = player_0_score + player_0_win_prob * sum(score_deck)
    # player_1_expected_score = player_1_score + player_1_win_prob * sum(score_deck)

    # # Create a dictionary of intermediate values
    # intermediate_values = {'player_0_win_prob': player_0_win_prob, 'player_1_win_prob': player_1_win_prob}

    # return (player_0_expected_score, player_1_expected_score), intermediate_values""",
    # """def evaluate_state(state) -> tuple[tuple[float, float], dict]:
    # score_cards = state[0] # list
    # player_0_played_cards = state[1] # list
    # player_1_played_cards = state[2] # list
    # is_turn = state[3] # bool
    # player_0_score = state[4] # float or int
    # player_1_score = state[5] # float or int
    # score_deck = state[6] # set
    # player_0_hand = state[7] # set
    # player_1_hand = state[8] # set

    # # Calculate the total value of remaining cards for each player
    # player_0_card_value = sum(player_0_hand)
    # player_1_card_value = sum(player_1_hand)

    # # Calculate the card advantage for each player
    # card_advantage_0 = player_0_card_value - player_1_card_value
    # card_advantage_1 = player_1_card_value - player_0_card_value

    # # Initial expected scores based on current scores
    # player_0_expected_score = player_0_score
    # player_1_expected_score = player_1_score

    # # Adjust expected scores based on card advantage
    # player_0_expected_score += 0.5 * card_advantage_0
    # player_1_expected_score += 0.5 * card_advantage_1

    # # Implement non-linear weighting system based on card values
    # card_weights = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5, 6: 0.6, 7: 0.7, 8: 0.8, 9: 0.9, 10: 1.0}
    
    # # Custom non-linear weights for GOPS game
    # custom_card_weights = {1: 0.2, 2: 0.3, 3: 0.4, 4: 0.6, 5: 0.8, 6: 1.0, 7: 1.2, 8: 1.4, 9: 1.6, 10: 2.0}
    
    # weighted_card_advantage_0 = sum([custom_card_weights[card] for card in player_0_hand]) - sum([custom_card_weights[card] for card in player_1_hand])
    # weighted_card_advantage_1 = sum([custom_card_weights[card] for card in player_1_hand]) - sum([custom_card_weights[card] for card in player_0_hand])

    # # Adjust expected scores based on weighted card advantage
    # player_0_expected_score += 0.3 * weighted_card_advantage_0
    # player_1_expected_score += 0.3 * weighted_card_advantage_1

    # # Create a dictionary of intermediate values for card advantage
    # intermediate_values = {'card_advantage_0': card_advantage_0, 'card_advantage_1': card_advantage_1,
    #                        'weighted_card_advantage_0': weighted_card_advantage_0, 'weighted_card_advantage_1': weighted_card_advantage_1}

    # return (player_0_expected_score, player_1_expected_score), intermediate_values""",
    # """def evaluate_state(state) -> tuple[tuple[float, float], dict]:
    # score_cards = state[0]
    # player_0_played_cards = state[1]
    # player_1_played_cards = state[2]
    # is_turn = state[3]
    # player_0_score = state[4]
    # player_1_score = state[5]
    # score_deck = state[6]
    # player_0_hand = state[7]
    # player_1_hand = state[8]
    
    # # Function to calculate the probability of winning a round based on the cards played and remaining cards
    # def calculate_round_win_probability(player_hand, opponent_played_cards, score_deck):
    #     total_rounds = len(score_deck)
    #     win_probability = 0
        
    #     if total_rounds == 0:
    #         return 0
        
    #     for card in player_hand:
    #         win_count = sum(card > opp_card for opp_card in opponent_played_cards)
    #         remaining_cards = [card for card in score_deck if card not in opponent_played_cards]
    #         for rem_card in remaining_cards:
    #             if card > rem_card:
    #                 win_count += 1
    #         win_probability += win_count / total_rounds
        
    #     return win_probability
    
    # # Calculate the probability of winning a round for each player based on their played cards and remaining cards
    # win_prob_player_0 = calculate_round_win_probability(player_0_hand, player_1_played_cards, score_deck)
    # win_prob_player_1 = calculate_round_win_probability(player_1_hand, player_0_played_cards, score_deck)
    
    # # Adjust expected scores based on round win probabilities
    # player_0_expected_score = player_0_score + win_prob_player_0 * sum(score_deck)
    # player_1_expected_score = player_1_score + win_prob_player_1 * sum(score_deck)
    
    # # Create a dictionary of intermediate values including the round win probabilities
    # intermediate_values = {'win_prob_player_0': win_prob_player_0, 'win_prob_player_1': win_prob_player_1}
    
    # return (player_0_expected_score, player_1_expected_score), intermediate_values"""
]

func_old_new = [
    # from jl
    """def evaluate_state(state) -> tuple[tuple[float, float], dict]:
    score_cards = state[0] 
    player_0_played_cards = state[1] 
    player_1_played_cards = state[2] 
    is_turn = state[3] 
    player_0_score = state[4] 
    player_1_score = state[5] 
    score_deck = state[6] 
    player_0_hand = state[7] 
    player_1_hand = state[8] 
    
    def calculate_total_score(player_played_cards):
        return sum(player_played_cards)
    
    def check_consecutive_rounds(player_played_cards):
        consecutive_rounds = 0
        for i in range(len(player_played_cards) - 1):
            if player_played_cards[i] > player_played_cards[i + 1]:
                consecutive_rounds += 1
        return consecutive_rounds
    
    player_0_total_score = calculate_total_score(player_0_played_cards)
    player_1_total_score = calculate_total_score(player_1_played_cards)
    
    score_difference = player_0_total_score - player_1_total_score
    
    remaining_score_cards = len(score_deck)
    
    # Calculate potential score for each player based on the distribution of card values in their hands
    def calculate_weighted_score(player_hand):
        weighted_score = sum([card if card > 2 else card * 0.5 for card in player_hand])
        return weighted_score
    
    player_0_weighted_score = calculate_weighted_score(player_0_hand)
    player_1_weighted_score = calculate_weighted_score(player_1_hand)
    
    player_0_consecutive_rounds = check_consecutive_rounds(player_0_played_cards)
    player_1_consecutive_rounds = check_consecutive_rounds(player_1_played_cards)
    
    # New improvement: consider strategic options based on card combinations
    player_0_strategic_value = len(player_0_hand) * 0.5  # Placeholder strategic value for player 0
    player_1_strategic_value = len(player_1_hand) * 0.5  # Placeholder strategic value for player 1
    
    # Introducing a more advanced strategic value calculation
    def calculate_strategic_value(player_hand, opponent_hand, score_cards_remaining):
        # Simulate potential future rounds to determine the likelihood of winning given the current state
        # This could involve more complex analysis of card combinations and possible outcomes
        strategic_value = len(player_hand) - len(opponent_hand) + score_cards_remaining
        return strategic_value
    
    player_0_strategic_value = calculate_strategic_value(player_0_hand, player_1_hand, remaining_score_cards)
    player_1_strategic_value = calculate_strategic_value(player_1_hand, player_0_hand, remaining_score_cards)
    
    state_value = score_difference + remaining_score_cards + player_0_weighted_score - player_1_weighted_score + player_0_consecutive_rounds - player_1_consecutive_rounds + player_0_strategic_value - player_1_strategic_value
    
    return (player_0_score + state_value, player_1_score - state_value), {'score_difference': score_difference, 'remaining_score_cards': remaining_score_cards, 'player_0_weighted_score': player_0_weighted_score, 'player_1_weighted_score': player_1_weighted_score, 'player_0_consecutive_rounds': player_0_consecutive_rounds, 'player_1_consecutive_rounds': player_1_consecutive_rounds, 'player_0_strategic_value': player_0_strategic_value, 'player_1_strategic_value': player_1_strategic_value}"""
]