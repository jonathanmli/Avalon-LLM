avalon_func_list = [
"""def evaluate_state(state: dict) -> tuple[dict[int, float], dict]:
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

    prob_good_victory = 0.5

    if phase == 0:  # Team selection phase
        # Placeholder values for team selection phase
        intermediate_value1 = 0.0
        intermediate_value2 = 0.0

    elif phase == 1:  # Team approval phase
        # Placeholder values for team approval phase
        intermediate_value1 = 0.0
        intermediate_value2 = 0.0

    elif phase == 2:  # Quest phase
        successful_quests = sum(historical_quest_results)
        failed_quests = len(historical_quest_results) - successful_quests
        if successful_quests > failed_quests:
            prob_good_victory += 0.2
        else:
            prob_good_victory -= 0.2

        intermediate_value1 = successful_quests
        intermediate_value2 = failed_quests

    elif phase == 3:  # Assassination phase
        # Evaluate Assassin's probability of successfully identifying and assassinating Merlin
        assassin_index = roles.index('Assassin')
        merlin_index = roles.index('Merlin')
        assassin_is_good = is_good[assassin_index]
        merlin_is_good = is_good[merlin_index]

        if not assassin_is_good:
            if merlin_is_good:
                # Assassin is Evil and Merlin is Good, so Assassin can assassinate Merlin
                prob_good_victory -= 0.2
            else:
                # Assassin is Evil but Merlin is Evil or another role, so no chance to assassinate
                prob_good_victory += 0.2

        intermediate_value1 = 0.0  # Placeholder for intermediate value 1
        intermediate_value2 = 0.0  # Placeholder for intermediate value 2

    expected_winrates_per_player = {}
    prob_evil_victory = 1 - prob_good_victory
    for player in players:
        if is_good[player]:
            expected_winrates_per_player[player] = prob_good_victory
        else:
            expected_winrates_per_player[player] = prob_evil_victory

    intermediate_values = {'assassination_probability': prob_good_victory}
    return expected_winrates_per_player, intermediate_values""",
"""def evaluate_state(state: dict) -> tuple[dict[Any, float], dict]:
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

    prob_good_victory = 0.5  # probability that Good will win

    if phase == 0:  # team selection phase
        # Placeholder values for intermediate calculations
        intermediate_value1 = 0
        intermediate_value2 = 0

    elif phase == 1:  # team approval phase
        # Placeholder values for intermediate calculations
        intermediate_value1 = 0
        intermediate_value2 = 0

    elif phase == 2:  # quest phase
        # Placeholder values for intermediate calculations
        intermediate_value1 = 0
        intermediate_value2 = 0

    elif phase == 3:  # assassination phase
        # Calculate the probability of Merlin being identified by the Evil players
        merlin_index = roles.index('Merlin')
        assassin_index = roles.index('Assassin')
        is_merlin_assassin_target = (assassin_index in quest_team)

        # Calculate the probability of Merlin being assassinated
        prob_merlin_assassinated = 0
        if is_merlin_assassin_target:
            num_failed_quests = len([result for result in historical_quest_results if not result])
            num_failed_required = 3 - num_failed_quests
            prob_merlin_assassinated = num_failed_required / (5 - turn)  # Adjust for remaining quests

        intermediate_value1 = prob_merlin_assassinated
        intermediate_value2 = 0  # Placeholder value

    expected_winrates_per_player = dict()
    prob_evil_victory = 1 - prob_good_victory
    for player in players:
        if is_good[player]:
            expected_winrates_per_player[player] = prob_good_victory
        else:
            expected_winrates_per_player[player] = prob_evil_victory

    intermediate_values = {'merlin_assassination_probability': intermediate_value1}
    player_scores = expected_winrates_per_player
    return player_scores, intermediate_values""",
"""def evaluate_state(state: dict) -> tuple[dict[Any, float], dict]:
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

    prob_good_victory = 0.5

    intermediate_values = {}

    if phase == 0:  # Team Selection Phase
        # Analyze team selection patterns or behaviors here
        # Update prob_good_victory accordingly
        intermediate_values['team_selection_analysis'] = "Placeholder analysis"
    
    elif phase == 1:  # Team Approval Phase
        # Analyze team approval voting patterns here
        last_team_rejected = len(historical_team_votes) > 0 and not historical_team_votes[-1]

        if last_team_rejected:
            prob_good_victory = 0.4
        else:
            prob_good_victory = 0.5

    elif phase == 2:  # Quest Phase
        num_successful_quests = historical_quest_results.count(True)
        num_failed_quests = historical_quest_results.count(False)

        if num_successful_quests > num_failed_quests:
            prob_good_victory = 0.6
        else:
            prob_good_victory = 0.5

        # Role-specific calculations for Quest Phase
        merlin_index = roles.index('Merlin')
        assassin_index = roles.index('Assassin')
        merlin_on_quest = merlin_index in quest_team
        assassin_on_quest = assassin_index in quest_team

        if merlin_on_quest and not assassin_on_quest:
            prob_good_victory += 0.1
        elif assassin_on_quest:
            prob_good_victory -= 0.1

        intermediate_values['merlin_on_quest'] = merlin_on_quest
        intermediate_values['assassin_on_quest'] = assassin_on_quest

    elif phase == 3:  # Assassination Phase
        merlin_index = roles.index('Merlin')
        merlin_targeted = merlin_index in quest_team

        prob_merlin_assassinated = 0.5
        if merlin_targeted:
            evil_players_indices = [i for i, role in enumerate(roles) if not is_good[i]]
            evil_players_on_team = [player for player in quest_team if player in evil_players_indices]
            prob_merlin_assassinated = 0.5 + (0.1 * len(evil_players_on_team))

        intermediate_values['merlin_targeted'] = merlin_targeted
        intermediate_values['prob_merlin_assassinated'] = prob_merlin_assassinated

    prob_evil_victory = 1 - prob_good_victory

    expected_winrates_per_player = {}
    for player in players:
        if is_good[player]:
            expected_winrates_per_player[player] = prob_good_victory
        else:
            expected_winrates_per_player[player] = prob_evil_victory

    intermediate_values['prob_good_victory'] = prob_good_victory

    return expected_winrates_per_player, intermediate_values""",
"""def evaluate_state(state: dict) -> tuple[dict[Any, float], dict]:
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

    prob_good_victory = 0.5  # Initial probability that Good will win

    if phase == 0:  # Team Selection Phase
        # Placeholder values for illustration, you should replace these with your own calculations
        intermediate_value1 = 0.1
        intermediate_value2 = 0.2

    elif phase == 1:  # Team Approval Phase
        # Placeholder values for illustration, you should replace these with your own calculations
        intermediate_value1 = 0.3
        intermediate_value2 = 0.4

    elif phase == 2:  # Quest Phase
        # Placeholder values for illustration, you should replace these with your own calculations
        intermediate_value1 = 0.5
        intermediate_value2 = 0.6

    elif phase == 3:  # Assassination Phase
        # Placeholder values for illustration, you should replace these with your own calculations
        intermediate_value1 = 0.7
        intermediate_value2 = 0.8

    # Calculate the expected win rates for each player based on the current phase
    expected_winrates_per_player = dict()
    prob_evil_victory = 1 - prob_good_victory
    for player in players:
        if is_good[player]:
            expected_winrates_per_player[player] = prob_good_victory
        else:
            expected_winrates_per_player[player] = prob_evil_victory

    # Store any important intermediate values used in the calculations
    intermediate_values = {'intermediate_value1': intermediate_value1, 'intermediate_value2': intermediate_value2}

    return expected_winrates_per_player, intermediate_values  # Return the player win rates and intermediate values""",
"""def evaluate_state(state: dict) -> tuple[dict[int, float], dict]:
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

    prob_good_victory = 0.5  # Initialize the probability that Good will win

    intermediate_values = {}  # Initialize intermediate values dictionary

    if phase == 0:  # Team selection phase
        # Add logic here to adjust win rates based on team selection phase
        if 'Merlin' in roles:
            prob_good_victory += 0.1
        if roles[quest_leader] in ['Minion', 'Assassin']:
            prob_good_victory -= 0.1
        else:
            prob_good_victory += 0.1

    elif phase == 1:  # Team approval phase
        # Add logic here to adjust win rates based on team approval phase
        if len(historical_team_votes) >= 4:
            prob_good_victory -= 0.1
        else:
            if 'Merlin' in roles:
                prob_good_victory += 0.1
            if roles[quest_leader] in ['Minion', 'Assassin']:
                prob_good_victory -= 0.1
            else:
                prob_good_victory += 0.1

    elif phase == 2:  # Quest phase
        # Add logic here to adjust win rates based on quest phase
        successful_quests = sum(historical_quest_results)
        failed_quests = len(historical_quest_results) - successful_quests
        if 'Merlin' in roles:
            if 'Assassin' in roles:
                prob_merlin_assassination = 0.5  # Probability of Assassin successfully assassinating Merlin
                prob_evil_win_merlin_assassinated = 1 - prob_merlin_assassination
                prob_evil_win_three_fails = 0.5  # Assume equal chance of Evil winning by three fails
                intermediate_values['prob_merlin_assassination'] = prob_merlin_assassination
                intermediate_values['prob_evil_win_merlin_assassinated'] = prob_evil_win_merlin_assassinated
                intermediate_values['prob_evil_win_three_fails'] = prob_evil_win_three_fails
                prob_good_victory = prob_good_victory * (1 - prob_merlin_assassination) * (1 - prob_evil_win_three_fails)
            else:
                prob_good_victory -= 0.2

        if successful_quests > failed_quests:
            prob_good_victory += 0.2
        else:
            prob_good_victory -= 0.2

    elif phase == 3:  # Assassination phase
        # Add logic here to adjust win rates based on assassination phase
        if 'Merlin' in roles:
            prob_good_victory -= 0.2
        else:
            prob_good_victory += 0.2

    # Adjust win rates based on information asymmetry between Good and Evil players
    for player in players:
        if not is_good[player]:  # Evil player
            prob_good_victory -= 0.1

    # Calculate the expected win rate for each player
    expected_winrates_per_player = {}
    prob_evil_victory = 1 - prob_good_victory
    for player in players:
        if is_good[player]:
            expected_winrates_per_player[player] = prob_good_victory
        else:
            expected_winrates_per_player[player] = prob_evil_victory

    return expected_winrates_per_player, intermediate_values""",
"""def evaluate_state(state: dict) -> tuple[dict[int, float], dict]:
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

    prob_good_victory = 0.5

    intermediate_value_team_selection = 0
    intermediate_value_approval_phase = 0
    intermediate_value_quest_phase = 0
    intermediate_value_assassination_phase = 0

    if phase == 0: 
        prob_good_victory = 0.5
        intermediate_value_team_selection = 0

    elif phase == 1: 
        suspicious_voters = set()
        for i, vote in enumerate(historical_team_votes):
            if vote == False:
                suspicious_voters.add(i)
        
        num_suspicious_voters = len(suspicious_voters)
        prob_good_victory += 0.05 * num_suspicious_voters
        
        intermediate_value_approval_phase = num_suspicious_voters

    elif phase == 2: 
        num_successful_quests = sum(historical_quest_results)
        num_failed_quests = len(historical_quest_results) - num_successful_quests

        impact_on_quest = 0
        for i, role in enumerate(roles):
            if role == 'Minion' or role == 'Assassin':
                if num_failed_quests >= 3:  
                    impact_on_quest += 0.3
                else:
                    impact_on_quest -= 0.3

        prob_good_victory += impact_on_quest
        
        intermediate_value_quest_phase = num_successful_quests

    elif phase == 3: 
        prob_merlin_assassination = 0
        if 'Merlin' in roles:
            prob_merlin_assassination = 1 / num_good
        prob_good_victory -= prob_merlin_assassination
        
        intermediate_value_assassination_phase = prob_merlin_assassination

    expected_winrates_per_player = dict()
    prob_evil_victory = 1 - prob_good_victory
    for player in players:
        if is_good[player]:
            expected_winrates_per_player[player] = prob_good_victory
        else:
            expected_winrates_per_player[player] = prob_evil_victory
    
    intermediate_values = {'team_selection_suspicious_voters': intermediate_value_team_selection,
                           'team_approval_num_suspicious_voters': intermediate_value_approval_phase,
                           'quest_successful_count': intermediate_value_quest_phase,
                           'merlin_assassination_probability': intermediate_value_assassination_phase}
    
    return expected_winrates_per_player, intermediate_values""",
"""def evaluate_state(state: dict) -> tuple[dict[Any, float], dict]:
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

    prob_good_victory = 0.5  # probability that Good will win

    intermediate_values = {}

    if phase == 0:  # team selection phase
        pass

    elif phase == 1:  # team approval phase
        pass

    elif phase == 2:  # quest phase
        player_weights = {}
        for player in players:
            weight = 1.0  # Default weight
            if is_good[player]:
                weight += 0.2  # Good players may have a higher weight
            if player in quest_team:
                weight += 0.1  # Players in the quest team may have a higher weight
            if historical_quest_results and historical_team_votes:
                num_successful_quests = sum(historical_quest_results)
                num_failed_quests = len(historical_quest_results) - num_successful_quests
                quest_fail_rate = num_failed_quests / len(historical_quest_results)

                if quest_fail_rate > 0.5:
                    weight -= 0.1  # Adjust weight based on historical quest performance

                team_approval_rate = historical_team_votes.count(True) / len(historical_team_votes)
                if team_approval_rate > 0.5:
                    weight += 0.1  # Adjust weight based on historical team approval rate

            player_weights[player] = weight

        total_weight = sum(player_weights.values())

        expected_winrates_per_player = {player: prob_good_victory * (player_weights[player] / total_weight) for player in players}

        intermediate_values['player_weights'] = player_weights

    elif phase == 3:  # assassination phase
        assassin = [idx for idx, role in enumerate(roles) if role == 'Assassin'][0]
        merlin = [idx for idx, role in enumerate(roles) if role == 'Merlin'][0]

        if is_good[assassin] and is_good[merlin]:  # If both Assassin and Merlin are Good
            assassin_win_rate = 0.0
        elif not is_good[assassin] and not is_good[merlin]:  # If both Assassin and Merlin are Evil
            assassin_win_rate = 0.0
        elif not is_good[assassin] and is_good[merlin]:  # If Assassin is Evil and Merlin is Good
            assassin_win_rate = 1.0  # Assassin can win by assassinating Merlin
        elif is_good[assassin] and not is_good[merlin]:  # If Assassin is Good and Merlin is Evil
            assassin_win_rate = 0.0

        intermediate_values['assassin_win_rate'] = assassin_win_rate

    prob_evil_victory = 1 - prob_good_victory

    expected_winrates_per_player = {}
    for player in players:
        if is_good[player]:
            expected_winrates_per_player[player] = prob_good_victory
        else:
            expected_winrates_per_player[player] = prob_evil_victory

    player_scores = expected_winrates_per_player

    return player_scores, intermediate_values""",
"""def evaluate_state(state):
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

    return expected_winrates_per_player, intermediate_values""",
"""def evaluate_state(state: dict) -> tuple[dict[Any, float], dict]:
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

    prob_good_victory = 0.5

    intermediate_values = {}

    if phase == 0:  # team selection phase
        # Placeholder values for team selection phase
        # You can add specific role considerations here for team selection phase
        intermediate_values['team_selection_value'] = 0.1
        intermediate_values['team_selection_value2'] = 0.2

    elif phase == 1:  # team approval phase
        # Analyze voting patterns based on player roles during team approval phase
        # Good players are more likely to approve, Evil players may strategically reject
        good_approval_rate = 0.8
        evil_approval_rate = 0.2
        for player, role in enumerate(roles):
            if is_good[player]:
                intermediate_values[f'player_{player}_approval_rate'] = good_approval_rate
            else:
                intermediate_values[f'player_{player}_approval_rate'] = evil_approval_rate

    elif phase == 2:  # quest phase
        # Placeholder values for quest phase
        intermediate_values['quest_phase_value'] = 0.5
        intermediate_values['quest_phase_value2'] = 0.6

    elif phase == 3:  # assassination phase
        # Placeholder values for assassination phase
        intermediate_values['assassination_value'] = 0.7
        intermediate_values['assassination_value2'] = 0.8

    expected_winrates_per_player = dict()
    prob_evil_victory = 1 - prob_good_victory
    for player in players:
        if is_good[player]:
            expected_winrates_per_player[player] = prob_good_victory
        else:
            expected_winrates_per_player[player] = prob_evil_victory

    # Include any important intermediate values that were used to calculate the scores
    intermediate_values['<intermediate_value1>'] = 0.0
    intermediate_values['<intermediate_value2>'] = 0.0

    return expected_winrates_per_player, intermediate_values  # Return the player scores and intermediate values.""",
"""def evaluate_state(state: dict) -> tuple[dict[Any, float], dict]:
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

    prob_good_victory = 0.5  # Initial probability that Good will win

    intermediate_value1 = ''
    intermediate_value2 = ''

    if phase == 0:  # Team Selection Phase
        # No voting patterns to consider in this phase
        intermediate_value1 = 'Team_Selection_Phase'
        intermediate_value2 = 'No_voting_patterns_to_consider'

    elif phase == 1:  # Team Approval Phase
        # Analyze voting patterns during team approval phase
        if historical_team_votes:
            num_approve = historical_team_votes.count(True)
            num_reject = historical_team_votes.count(False)

            approve_rate = num_approve / (num_approve + num_reject)

            # Adjust probabilities based on voting patterns
            if approve_rate >= 0.5:  # High approval rate favors Good
                prob_good_victory = 0.6
            else:  # Low approval rate favors Evil
                prob_good_victory = 0.4

            # Analyze voting patterns and adjust win rates based on player behavior
            player_vote_patterns = {player: 0.5 for player in players}

            for idx, vote in enumerate(historical_team_votes):
                for player in players:
                    if vote != (not is_good[player]):
                        player_vote_patterns[player] -= 0.1

            # Calculate the overall good win rate based on players' voting patterns
            prob_good_victory = sum(player_vote_patterns[player] for player in players if is_good[player]) / num_good

            intermediate_value1 = 'Team_Approval_Phase'
            intermediate_value2 = 'Analyzing_voting_patterns'

    elif phase == 2:  # Quest Phase
        # Analyze quest results to improve heuristic
        total_quests = len(historical_quest_results)
        if total_quests == 0:  # Handle division by zero case
            prob_good_victory = 0.5
        else:
            # Initialize win rates for each player
            player_winrates = {player: 0.5 for player in players}

            for quest_result in historical_quest_results:
                if not quest_result:  # Quest failed
                    for player in quest_team:
                        if is_good[player]:
                            player_winrates[player] -= 0.1
                        else:
                            player_winrates[player] += 0.1

            # Calculate the overall good win rate based on players' win rates
            prob_good_victory = sum(player_winrates[player] for player in players if is_good[player]) / num_good

            intermediate_value1 = 'Quest_Phase'
            intermediate_value2 = 'Analyzing_quest_results'

    elif phase == 3:  # Assassination Phase
        # Example: Determine the probability of Merlin being assassinated
        merlin = roles.index('Merlin')
        if merlin in quest_team:
            prob_merlin_assassinated = 1.0

        intermediate_value1 = 'Assassination_Phase'
        intermediate_value2 = 'Analyzing_assassination'

    # Improved Voting Patterns Analysis
    def update_player_voting_strategies(player_vote_patterns, historical_team_votes, is_good):
        for idx, vote in enumerate(historical_team_votes):
            for player in players:
                if vote != (not is_good[player]):
                    if is_good[player]:
                        player_vote_patterns[player] -= 0.1
                    else:
                        player_vote_patterns[player] += 0.1
        return player_vote_patterns

    player_vote_patterns = {player: 0.5 for player in players}
    player_vote_patterns = update_player_voting_strategies(player_vote_patterns, historical_team_votes, is_good)

    # Calculate the overall good win rate based on players' updated voting patterns
    prob_good_victory = sum(player_vote_patterns[player] for player in players if is_good[player]) / num_good

    expected_winrates_per_player = dict()
    prob_evil_victory = 1 - prob_good_victory
    for player in players:
        if is_good[player]:
            expected_winrates_per_player[player] = prob_good_victory
        else:
            expected_winrates_per_player[player] = prob_evil_victory

    intermediate_values = {'intermediate_value1': intermediate_value1, 'intermediate_value2': intermediate_value2}
    player_scores = expected_winrates_per_player
    return player_scores, intermediate_values  # Final results in the required format"""
]