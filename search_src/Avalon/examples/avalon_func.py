# avalon_func_list = [
#     """def evaluate_state(state):
#     # Extracting state information
#     players = state['players']
#     turn = state['turn']
#     phase = state['phase']
#     round = state['round']
#     quest_leader = state['quest_leader']
#     quest_team = state['quest_team']
#     historical_quest_results = state['historical_quest_results']
#     historical_team_votes = state['historical_team_votes']
#     num_good = state['num_good']
#     num_evil = len(players) - num_good
#     num_participants_per_quest = state['num_participants_per_quest']
#     num_fails_per_quest = state['num_fails_per_quest']

#     # Initialising players' expected win rates
#     player_rates = {p: 0.5 for p in players} # Since we don't have any information about roles yet

#     # Calculating intermediate values:
#     total_quests = len(num_participants_per_quest)
#     total_rounds = 5 # In every turn, there can be at most 5 rounds of discussions
#     quests_passed = sum(historical_quest_results) 
#     quests_failed = turn - quests_passed # Subtract quests passed from total quests attempted
#     rounds_failed = sum([1 for vote in historical_team_votes if vote is False])

#     if quests_passed == 3: # 3 Successful Quests for Good Team => Good team wins
#         player_rates = {player: (1 if player in quest_team else 0) for player in players}
#     elif quests_failed == 3: # 3 Successful Quests for Evil Team => Evil team wins
#         player_rates = {player: (1 if player not in quest_team else 0) for player in players}
#     else:
#         # More successful quests increase win rates for the quest team
#         for player in quest_team:
#             player_rates[player] += quests_passed / total_quests
        
#         # More failed quests increase win rates for the non-quest team
#         for player in players - quest_team:
#             player_rates[player] += quests_failed / total_quests
   
#     # Calculate extra score for non-rejected rounds
#     for qb in historical_team_votes:
#         if qb: # If a quest team is approved, increase win rate for quest team members
#             for player in quest_team:
#                 player_rates[player]+= 0.1
#         else: # If it is rejected, increase win rate for non-quest team members
#             for player in players - quest_team:
#                 player_rates[player]+= 0.1

    
#     intermediate_values = {'turn' : turn,
#                             'quests_passed' : quests_passed,
#                             'quests_failed' : quests_failed,
#                             'rounds_failed' : rounds_failed}
    
#     return player_rates, intermediate_values"""
# ]

avalon_func_list = [
"""def evaluate_state(state: dict) -> tuple[dict[Any, float], dict]:
    # Extract relevant information from the game state
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
    
    # Placeholder for intermediate values
    intermediate_values = {}
    
    # Calculate expected win rate for each player
    player_win_rates = {}
    
    # Sample calculation of win rates - replace with your own logic
    for player in players:
        if is_good[player]:
            # Example: Good players have a higher win rate
            player_win_rates[player] = 0.7
        else:
            # Example: Evil players have a lower win rate
            player_win_rates[player] = 0.3
    
    # Add intermediate values to the dictionary
    intermediate_values['example_value'] = 0.5
    
    return player_win_rates, intermediate_values""",
    """def evaluate_state(state: dict) -> tuple[dict[Any, float], dict]:
    # Helper function to calculate the likelihood of Merlin being assassinated
    def calculate_merlin_assassination_probability(state):
        # Dummy implementation for now
        return 0.0

    # Helper function to evaluate the team composition
    def evaluate_team_composition(state):
        # Dummy implementation for now
        return {}

    # Helper function to evaluate the voting patterns
    def evaluate_voting_patterns(state):
        # Dummy implementation for now
        return {}

    # Helper function to evaluate the claims credibility
    def evaluate_claims_credibility(state):
        # Dummy implementation for now
        return {}

    # Helper function to evaluate the strategic advantage
    def evaluate_strategic_advantage(state):
        # Dummy implementation for now
        return 0.0

    # Helper function to calculate the number of Quests succeeded by the Good players
    def calculate_quests_succeeded(state, side):
        # Dummy implementation for now
        return 0

    # Helper function to calculate the number of Quests failed by the Evil players
    def calculate_quests_failed(state, side):
        # Dummy implementation for now
        return 0

    # Helper function to check if Merlin has been identified by the Evil players
    def check_merlin_identified(state):
        # Dummy implementation for now
        return False

    # Calculate the number of Quests succeeded by the Good players and failed by the Evil players
    quests_succeeded_by_good = calculate_quests_succeeded(state, "Good")
    quests_failed_by_evil = calculate_quests_failed(state, "Evil")

    # Check if Merlin has been identified by the Evil players
    merlin_identified = check_merlin_identified(state)

    # Evaluate team composition, voting patterns, claims credibility, and strategic advantage
    team_composition = evaluate_team_composition(state)
    voting_patterns = evaluate_voting_patterns(state)
    claims_credibility = evaluate_claims_credibility(state)
    strategic_advantage = evaluate_strategic_advantage(state)

    # Calculate the likelihood of Good winning the game
    likelihood_of_good_win = 0.5  # Placeholder value for now

    # Construct the dictionary of player expected win rates
    player_expected_winrates = {player_id: 0.5 for player_id in state['players']}  # Initialize all players with 0.5 chance of winning

    # Construct the dictionary of intermediate values
    intermediate_values = {
        'merlin_assassination_probability': calculate_merlin_assassination_probability(state),
        'team_composition': team_composition,
        'voting_patterns': voting_patterns,
        'claims_credibility': claims_credibility,
        'strategic_advantage': strategic_advantage
    }

    return player_expected_winrates, intermediate_values""",
    """def evaluate_state(state: dict) -> tuple[dict[Any, float], dict]:
    # Helper function to calculate the win rate of a player based on their role
    def calculate_win_rate(role, is_good):
        if role == 'Merlin':
            return 0.8 if is_good else 0.2
        elif role == 'Assassin':
            return 0.2 if is_good else 0.8
        else:
            return 0.5

    players = state['players']
    roles = state['roles']
    is_good = state['is_good']

    # Calculate the expected win rate for each player based on their role
    player_winrates = {player: calculate_win_rate(roles[player], is_good[player]) for player in players}

    # Initialize the intermediate values dictionary
    intermediate_values = {}

    # You can add more intermediate calculations here as needed

    return player_winrates, intermediate_values"""
]

avalon_best_functions = ["""def evaluate_state(state: dict) -> tuple[dict[Any, float], dict]:
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