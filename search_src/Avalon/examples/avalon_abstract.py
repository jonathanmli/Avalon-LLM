avalon_abstract_list = [
"""Thoughts:
In The Resistance: Avalon, the value of a state can be determined by several key elements:
- The number of Quests that have succeeded or failed: This indicates the progress of each team towards their respective goals.
- The voting patterns of players: This can provide insights into potential alliances, trust, and deception among players.
- The choices made by players during the Quests: Passing or failing a Quest can reveal the intentions and strategies of players.

Pseudocode:
1. Define a function evaluate_state(state) that takes the current state of the game as input.
2. Calculate the number of Quests that have succeeded and failed in the current state.
3. Analyze the voting patterns of players to identify any trends or suspicious behavior.
4. Evaluate the choices made by players during the Quests to determine the likelihood of a player being Good or Evil.
5. Consider the roles of Merlin and Assassin, as their identities can significantly impact the outcome of the game.
6. Assign a value to the state based on the above factors, giving more weight to information that is more indicative of a player's true intentions.
7. Return the value of the state as an indicator of the current state of the game and the potential outcomes for each team.

Pseudocode:
```
function evaluate_state(state):
    num_quests_succeeded = calculate_num_quests_succeeded(state)
    num_quests_failed = calculate_num_quests_failed(state)
    
    voting_patterns = analyze_voting_patterns(state)
    player_choices = evaluate_player_choices(state)
    
    merlin_found = is_merlin_exposed(state)
    assassin_alive = is_assassin_alive(state)
    
    state_value = assign_value(num_quests_succeeded, num_quests_failed, voting_patterns, player_choices, merlin_found, assassin_alive)
    
    return state_value
```""",
"""Thoughts:
In The Resistance: Avalon, the value of a state can be determined by the following factors:
- The number of Quests succeeded by the Good players.
- The number of Quests failed by the Evil players.
- The identity of Merlin and whether they have been identified by the Evil players.
- The state of discussion and deception among players.
- The composition of teams selected for each Quest.
- The voting patterns of players during team selection.
- The claims made by different players and their credibility.

Pseudocode:
1. Define a function evaluate_state(state) that takes the current state of the game as input.
2. Calculate the number of Quests succeeded by the Good players and failed by the Evil players based on the current state.
3. Check if Merlin has been identified by the Evil players.
4. Evaluate the state based on the composition of teams selected for each Quest, the voting patterns, claims made by players, and the overall deception in the game.
5. Consider the strategic advantage/disadvantage for each team based on the current state.
6. Return a value heuristic that provides an estimate of the likelihood of Good or Evil winning the game based on the current state.

Pseudocode:
```
function evaluate_state(state):
    quests_succeeded_by_good = calculate_quests_succeeded(state, "Good")
    quests_failed_by_evil = calculate_quests_failed(state, "Evil")
    
    merlin_identified = check_merlin_identified(state)
    
    team_composition = evaluate_team_composition(state)
    voting_patterns = evaluate_voting_patterns(state)
    claims_credibility = evaluate_claims_credibility(state)
    
    strategic_advantage = evaluate_strategic_advantage(state)
    
    likelihood_of_good_win = calculate_likelihood_of_good_win(quests_succeeded_by_good, quests_failed_by_evil, merlin_identified, team_composition, voting_patterns, claims_credibility, strategic_advantage)
    
    return likelihood_of_good_win
```""",
"""Thoughts:
In The Resistance: Avalon, the value of a state can be determined by the current progress of the game towards either the Good or Evil side. The key factors that can help evaluate a state include:
- The number of Quests that have succeeded or failed.
- The players' claims, discussions, and accusations.
- The number of teams that have been rejected in a row.
- The knowledge and actions of Merlin and the Assassin.
- The voting patterns of players on teams and Quests.

Pseudocode:
1. Define a function evaluate_state(state) that takes the current state of the game as input.
2. Calculate the progress towards the winning conditions for both the Good and Evil sides based on the current state.
3. Consider the impact of team rejections and the possibility of the fifth team being automatically approved.
4. Analyze the voting history and patterns of players to assess trustworthiness and deception.
5. Evaluate the potential value of the state by considering factors such as:
   - The number of Quests that have succeeded or failed.
   - The claim and knowledge of Merlin and the Assassin.
   - The balance of power between the Good and Evil sides.
6. Return a tuple containing the expected value for the current state, considering both the Good and Evil sides.

Pseudocode:
```
function evaluate_state(state):
    good_progress = calculate_good_progress(state)
    evil_progress = calculate_evil_progress(state)
    
    team_rejections = get_team_rejections(state)
    automatic_approval = check_automatic_approval(team_rejections)
    
    voting_history = get_voting_history(state)
    trustworthiness = analyze_voting_patterns(voting_history)
    
    potential_value = calculate_potential_value(good_progress, evil_progress, automatic_approval, trustworthiness)
    
    return potential_value
```"""
]