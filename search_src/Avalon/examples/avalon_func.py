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
    
    return player_rates, intermediate_values"""
]