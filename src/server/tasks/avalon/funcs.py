avalon_func = """def evaluate_state(state):
    players = state['players']
    turn = state['turn']
    phase = state['phase']
    round_number = state['round']
    quest_leader = state['quest_leader']
    quest_team = state['quest_team']
    historical_quest_results = state['historical_quest_results']
    historical_team_votes = state['historical_team_votes']
    num_good = state['num_good']
    num_participants_per_quest = state['num_participants_per_quest']
    num_fails_per_quest = state['num_fails_per_quest']
    roles = state['roles']
    is_good = state['is_good']

    prob_good_victory = 0.5  # Initialize the probability that Good will win

    if phase == 0:  # Team selection phase
        if roles[quest_leader] in ['Minion', 'Assassin']:
            prob_good_victory -= 0.1
        else:
            prob_good_victory += 0.1

    elif phase == 1:  # Team approval phase
        if len(historical_team_votes) >= 4:
            prob_good_victory -= 0.1
        else:
            if roles[quest_leader] in ['Minion', 'Assassin']:
                prob_good_victory -= 0.1
            else:
                prob_good_victory += 0.1

    elif phase == 2:  # Quest phase
        successful_quests = sum(historical_quest_results)
        failed_quests = len(historical_quest_results) - successful_quests
        if successful_quests > failed_quests:
            prob_good_victory += 0.2
        else:
            prob_good_victory -= 0.2

    elif phase == 3:  # Assassination phase
        if 'Merlin' in roles:
            prob_good_victory -= 0.2
        else:
            prob_good_victory += 0.2

    # Calculate the expected win rate for each player
    expected_winrates_per_player = {}
    prob_evil_victory = 1 - prob_good_victory
    for player in players:
        if is_good[player]:
            expected_winrates_per_player[player] = prob_good_victory
        else:
            expected_winrates_per_player[player] = prob_evil_victory

    # Store any important intermediate values
    intermediate_values = {'prob_good_victory': prob_good_victory, 'prob_evil_victory': prob_evil_victory}

    return expected_winrates_per_player, intermediate_values"""