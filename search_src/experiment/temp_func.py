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
    
    # Revised calculation of potential scores for each player
    player_potential_score = calculate_potential_score(player_0_score, player_0_hand, remaining_score_cards)
    opponent_potential_score = calculate_potential_score(player_1_score, player_1_hand, remaining_score_cards)
    
    # Revised estimation of end game scores based on potential scores
    player_0_expected_score = player_0_score + player_potential_score
    player_1_expected_score = player_1_score + opponent_potential_score
    
    return (player_0_expected_score, player_1_expected_score), {'player_potential_score': player_potential_score, 'opponent_potential_score': opponent_potential_score}""",
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
    
    # Simulate future rounds to evaluate potential scores for both players
    def simulate_rounds(player_0_score, player_1_score, player_0_hand, player_1_hand, remaining_score_cards, round_num):
        if round_num == len(score_cards):
            return player_0_score, player_1_score
        
        new_score_cards = score_cards[round_num:]
        new_player_0_played_cards = player_0_played_cards[round_num:]
        new_player_1_played_cards = player_1_played_cards[round_num:]
        
        new_remaining_score_cards = set(remaining_score_cards)
        new_remaining_score_cards.difference_update(new_score_cards)
        
        # Simulate different scenarios based on possible card combinations
        for card_0 in player_0_hand:
            for card_1 in player_1_hand:
                new_player_0_score = player_0_score
                new_player_1_score = player_1_score
                
                if card_0 > card_1:
                    new_player_0_score += new_score_cards[0]
                elif card_1 > card_0:
                    new_player_1_score += new_score_cards[0]
                # If cards are equal, simulate next round to determine winner
                
                player_0_result, player_1_result = simulate_rounds(new_player_0_score, new_player_1_score, player_0_hand.difference({card_0}), player_1_hand.difference({card_1}), new_remaining_score_cards, round_num + 1)
                player_0_score = max(player_0_score, player_0_result)
                player_1_score = max(player_1_score, player_1_result)
                
        return player_0_score, player_1_score
    
    # Evaluate the potential value of the state by simulating future rounds
    player_0_expected_score, player_1_expected_score = simulate_rounds(player_0_score, player_1_score, player_0_hand, player_1_hand, remaining_score_cards, 0)
    
    return (player_0_expected_score, player_1_expected_score), {'player_0_expected_score': player_0_expected_score, 'player_1_expected_score': player_1_expected_score}""",
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
    
    # Modify the function to consider additional factors for more accurate evaluation
    # Add logic to account for strategic decisions, winning streaks, and card values
    
    # Example of an intermediate value calculation
    intermediate_value1 = player_potential_score - opponent_potential_score
    
    return (player_potential_score, opponent_potential_score), {'intermediate_value1': intermediate_value1}"""
]

avalon_func_list = [
    """def evaluate_state(state):
    # Extracting state information
    players = state['players']
    turn = state['turn']
    phase = state['phase']
    round = state['round']
    quest_leader = state['quest_leader']
    quest_team = state['quest_team']
    historical_quest_results = state['historical_quest_results']
    historical_team_votes = state['historical_team_votes']
    num_good = state['num_good']
    num_evil = len(players) - num_good
    num_participants_per_quest = state['num_participants_per_quest']
    num_fails_per_quest = state['num_fails_per_quest']

    # Initialising players' expected win rates
    player_rates = {p: 0.5 for p in players} # Since we don't have any information about roles yet

    # Calculating intermediate values:
    total_quests = len(num_participants_per_quest)
    total_rounds = 5 # In every turn, there can be at most 5 rounds of discussions
    quests_passed = sum(historical_quest_results) 
    quests_failed = turn - quests_passed # Subtract quests passed from total quests attempted
    rounds_failed = sum([1 for vote in historical_team_votes if vote is False])

    if quests_passed == 3: # 3 Successful Quests for Good Team => Good team wins
        player_rates = {player: (1 if player in quest_team else 0) for player in players}
    elif quests_failed == 3: # 3 Successful Quests for Evil Team => Evil team wins
        player_rates = {player: (1 if player not in quest_team else 0) for player in players}
    else:
        # More successful quests increase win rates for the quest team
        for player in quest_team:
            player_rates[player] += quests_passed / total_quests
        
        # More failed quests increase win rates for the non-quest team
        for player in players - quest_team:
            player_rates[player] += quests_failed / total_quests
   
    # Calculate extra score for non-rejected rounds
    for qb in historical_team_votes:
        if qb: # If a quest team is approved, increase win rate for quest team members
            for player in quest_team:
                player_rates[player]+= 0.1
        else: # If it is rejected, increase win rate for non-quest team members
            for player in players - quest_team:
                player_rates[player]+= 0.1

    
    intermediate_values = {'turn' : turn,
                            'quests_passed' : quests_passed,
                            'quests_failed' : quests_failed,
                            'rounds_failed' : rounds_failed}
    
    return player_rates, intermediate_values""",
    """def evaluate_state(state):
    # Extracting state information
    players = state['players']
    turn = state['turn']
    phase = state['phase']
    round = state['round']
    quest_leader = state['quest_leader']
    quest_team = state['quest_team']
    historical_quest_results = state['historical_quest_results']
    historical_team_votes = state['historical_team_votes']
    num_good = state['num_good']
    num_evil = len(players) - num_good
    num_participants_per_quest = state['num_participants_per_quest']
    num_fails_per_quest = state['num_fails_per_quest']

    # Initialising players' expected win rates
    player_rates = {p: 0.5 for p in players} # Since we don't have any information about roles yet

    # Calculating intermediate values:
    total_quests = len(num_participants_per_quest)
    total_rounds = 5 # In every turn, there can be at most 5 rounds of discussions
    quests_passed = sum(historical_quest_results) 
    quests_failed = turn - quests_passed # Subtract quests passed from total quests attempted
    rounds_failed = sum([1 for vote in historical_team_votes if vote is False])

    if quests_passed == 3: # 3 Successful Quests for Good Team => Good team wins
        player_rates = {player: (1 if player in quest_team else 0) for player in players}
    elif quests_failed == 3: # 3 Successful Quests for Evil Team => Evil team wins
        player_rates = {player: (1 if player not in quest_team else 0) for player in players}
    else:
        # More successful quests increase win rates for the quest team
        for player in quest_team:
            player_rates[player] += quests_passed / total_quests
        
        # More failed quests increase win rates for the non-quest team
        for player in players - quest_team:
            player_rates[player] += quests_failed / total_quests
   
    # Calculate extra score for non-rejected rounds
    for qb in historical_team_votes:
        if qb: # If a quest team is approved, increase win rate for quest team members
            for player in quest_team:
                player_rates[player]+= 0.1
        else: # If it is rejected, increase win rate for non-quest team members
            for player in players - quest_team:
                player_rates[player]+= 0.1

    
    intermediate_values = {'turn' : turn,
                            'quests_passed' : quests_passed,
                            'quests_failed' : quests_failed,
                            'rounds_failed' : rounds_failed}
    
    return player_rates, intermediate_values""",
    """def evaluate_state(state):
    # Extracting state information
    players = state['players']
    turn = state['turn']
    phase = state['phase']
    round = state['round']
    quest_leader = state['quest_leader']
    quest_team = state['quest_team']
    historical_quest_results = state['historical_quest_results']
    historical_team_votes = state['historical_team_votes']
    num_good = state['num_good']
    num_evil = len(players) - num_good
    num_participants_per_quest = state['num_participants_per_quest']
    num_fails_per_quest = state['num_fails_per_quest']

    # Initialising players' expected win rates
    player_rates = {p: 0.5 for p in players} # Since we don't have any information about roles yet

    # Calculating intermediate values:
    total_quests = len(num_participants_per_quest)
    total_rounds = 5 # In every turn, there can be at most 5 rounds of discussions
    quests_passed = sum(historical_quest_results) 
    quests_failed = turn - quests_passed # Subtract quests passed from total quests attempted
    rounds_failed = sum([1 for vote in historical_team_votes if vote is False])

    if quests_passed == 3: # 3 Successful Quests for Good Team => Good team wins
        player_rates = {player: (1 if player in quest_team else 0) for player in players}
    elif quests_failed == 3: # 3 Successful Quests for Evil Team => Evil team wins
        player_rates = {player: (1 if player not in quest_team else 0) for player in players}
    else:
        # More successful quests increase win rates for the quest team
        for player in quest_team:
            player_rates[player] += quests_passed / total_quests
        
        # More failed quests increase win rates for the non-quest team
        for player in players - quest_team:
            player_rates[player] += quests_failed / total_quests
   
    # Calculate extra score for non-rejected rounds
    for qb in historical_team_votes:
        if qb: # If a quest team is approved, increase win rate for quest team members
            for player in quest_team:
                player_rates[player]+= 0.1
        else: # If it is rejected, increase win rate for non-quest team members
            for player in players - quest_team:
                player_rates[player]+= 0.1

    
    intermediate_values = {'turn' : turn,
                            'quests_passed' : quests_passed,
                            'quests_failed' : quests_failed,
                            'rounds_failed' : rounds_failed}
    
    return player_rates, intermediate_values"""
]