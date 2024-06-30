functions = ["""def evaluate_state(state: dict) -> tuple[dict[Any, float], dict]:
    players: set[int] = state['players']
    turn: int = state['turn']
    phase: int = state['phase']
    round_number: int = state['round']
    quest_leader: int = state['quest_leader']
    quest_team: set[int] = state['quest_team']
    historical_quest_results: list[bool] = state['historical_quest_results']
    historical_team_votes: list[bool] = state['historical_team_votes']
    num_good: int = state['num_good']
    num_participants_per_quest: list[int] = state['num_participants_per_quest']
    num_fails_per_quest: list[int] = state['num_fails_per_quest']
    roles: list[str] = state['roles']
    is_good: list[bool] = state['is_good']

    # Initialize the probability that Good will win
    prob_good_victory = 0.5

    intermediate_values = {}

    if phase == 0:  # Team selection phase
        # Modify the probability of Good winning based on the role of the quest leader
        if roles[quest_leader] in ['Minion', 'Assassin']:
            prob_good_victory -= 0.1
        else:
            prob_good_victory += 0.1

        # Store the intermediate values used for calculations
        intermediate_values['role_influence'] = prob_good_victory

    elif phase == 1:  # Team approval phase
        if len(historical_team_votes) >= 4:
            prob_good_victory -= 0.1
        else:
            if roles[quest_leader] in ['Minion', 'Assassin']:
                prob_good_victory -= 0.1
            else:
                prob_good_victory += 0.1

        # Store the intermediate values used for calculations
        intermediate_values['approval_influence'] = prob_good_victory

    elif phase == 2:  # Quest phase
        successful_quests = sum(historical_quest_results)
        failed_quests = len(historical_quest_results) - successful_quests
        if successful_quests > failed_quests:
            prob_good_victory += 0.2
        else:
            prob_good_victory -= 0.2

        # Store the intermediate values used for calculations
        intermediate_values['quest_outcome'] = prob_good_victory

    elif phase == 3:  # Assassination phase
        if 'Merlin' in roles:
            prob_good_victory -= 0.2
        else:
            prob_good_victory += 0.2

        # Store the intermediate values used for calculations
        intermediate_values['assassination_outcome'] = prob_good_victory

    expected_winrates_per_player = {}
    prob_evil_victory = 1 - prob_good_victory
    for player in players:
        if is_good[player]:
            expected_winrates_per_player[player] = prob_good_victory
        else:
            expected_winrates_per_player[player] = prob_evil_victory

    return expected_winrates_per_player, intermediate_values""",
    """def evaluate_state(state: dict) -> tuple[dict[Any, float], dict]:
    players = state['players']
    turn = state['turn']
    phase = state['phase']
    round = state['round']
    quest_leader = state['quest_leader']
    quest_team = state['quest_team']
    historical_quest_results = state['historical_quest_results']
    historical_team_votes = state['historical_team_votes']
    num_good = state['num_good']
    num_participants_per_quest = state['num_participants_per_quest']
    num_fails_per_quest = state['num_fails_per_quest']
    roles = state['roles']
    is_good = state['is_good']

    prob_good_victory = 0.5  # probability that Good will win

    intermediate_values = {}

    if phase == 0:  # team selection phase
        # No need to adjust win rates in team selection phase
        intermediate_values['team_selection_value'] = 0.0

    elif phase == 1:  # team approval phase
        # Adjust win rates based on Evil players' knowledge
        expected_winrates_per_player = {}
        for player in players:
            if not is_good[player]:  # If player is Evil
                evil_players = [p for p in players if not is_good[p]]
                minions = [p for p in evil_players if roles[p] == 'Minion']
                # If player is a Minion, adjust their win rate based on the number of Minions in the game
                if roles[player] == 'Minion':
                    expected_winrates_per_player[player] = prob_good_victory / len(minions)
                else:
                    expected_winrates_per_player[player] = prob_good_victory / (len(evil_players) - len(minions))

        intermediate_values['team_approval_value'] = 0.0

    elif phase == 2:  # quest phase
        # No need to adjust win rates in quest phase
        intermediate_values['quest_phase_value'] = 0.0

    elif phase == 3:  # assassination phase
        # Adjust win rates based on the probability of Merlin being assassinated by Assassin
        expected_winrates_per_player = {}
        merlin = [p for p in players if roles[p] == 'Merlin'][0]
        assassin = [p for p in players if roles[p] == 'Assassin'][0]
        if turn == 5:  # Last turn, Assassin has a chance to assassinate Merlin
            if assassin in quest_team and 'fail' in quest_team[assassin]:
                # Merlin was on the quest team and failed the quest, so Assassin can assassinate Merlin
                expected_winrates_per_player[merlin] = 0.0
            else:
                # Merlin was not on the quest team or did not fail the quest, so Assassin cannot assassinate Merlin
                expected_winrates_per_player[merlin] = prob_good_victory

        intermediate_values['assassination_value'] = 0.0

    expected_winrates_per_player = {}
    prob_evil_victory = 1 - prob_good_victory
    for player in players:
        if is_good[player]:
            expected_winrates_per_player[player] = prob_good_victory
        else:
            expected_winrates_per_player[player] = prob_evil_victory

    return expected_winrates_per_player, intermediate_values""",]