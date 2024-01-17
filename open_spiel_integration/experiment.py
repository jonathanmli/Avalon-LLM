import pyspiel
from open_spiel_integration.open_spiel_bots import *

def run_gops_experiment(game, bots, num_episodes=100, rng=None):
    """Run a number of episodes of the game, and print statistics."""

    bot1 = bots[0]
    bot2 = bots[1]

    bot1_cumulative_return = 0
    bot2_cumulative_return = 0

    bot1_wins = 0
    bot2_wins = 0

    bot1_is_custom = isinstance(bot1, CustomBot)
    bot2_is_custom = isinstance(bot2, CustomBot)

    # assert that bots are either CustomBots or OpenSpielBots
    assert bot1_is_custom or isinstance(bot1, OpenSpielBot)
    assert bot2_is_custom or isinstance(bot2, OpenSpielBot)

    # add rng 
    if rng is None:
        rng = np.random.RandomState()

    # Play num_episodes games
    for i in range(num_episodes):
        # Run a game
        state = game.new_initial_state()
        num_cards = len(state.legal_actions())
        prize_cards = []
        player_cards = []
        opponent_cards = []

        while not state.is_terminal():
            if state.is_chance_node():
                # Sample a chance event outcome.
                outcomes_with_probs = state.chance_outcomes()
                action_list, prob_list = zip(*outcomes_with_probs)
                action = rng.choice(action_list, p=prob_list)
                state.apply_action(action)
                prize_cards.append(action+1)
            else:
                if state.current_player() == 0:
                    # player 1's turn
                    if bot1_is_custom:
                        gops_state = GOPSState('simultaneous', prize_cards, player_cards, opponent_cards, num_cards)
                        action = bot1.step(gops_state)
                    else:
                        action = bot1.step(state)
                    player_cards.append(action+1)
                else:
                    # player 2's turn
                    if bot2_is_custom:
                        gops_state = GOPSState('simultaneous', prize_cards, player_cards, opponent_cards, num_cards)
                        action = bot2.step(gops_state)
                    else:
                        action = bot2.step(state)
                    opponent_cards.append(action+1)

                state.apply_action(action)

        gops_state = GOPSState('dummy', prize_cards, player_cards, opponent_cards, num_cards, True)

        # Episode is over, update return
        # returns = state.returns()
        returns = gops_state.calculate_score()
        bot1_cumulative_return += returns[0]
        bot2_cumulative_return += returns[1]
        if returns[0] > returns[1]:
            bot1_wins += 1
        else:
            bot2_wins += 1

    # Print total returns from the viewpoint of player 1 and 2
    print("Player 1: {}".format(bot1_cumulative_return / num_episodes))
    print("Player 2: {}".format(bot2_cumulative_return / num_episodes))

    # Print winrate of player 1 and 2
    print("Player 1 winrate: {}".format(bot1_wins / num_episodes))
    print("Player 2 winrate: {}".format(bot2_wins / num_episodes))

    # returning the cumulative returns of player 1 and 2 and their winrates
    return bot1_cumulative_return / num_episodes, bot2_cumulative_return / num_episodes, bot1_wins / num_episodes, bot2_wins / num_episodes

def play_game(game, bots, rng=None):
    """Plays one game. Prints out states and actions as it goes along for debugging."""
    bot1 = bots[0]
    bot2 = bots[1]

    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            # Sample a chance event outcome.
            outcomes_with_probs = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            action = rng.choice(action_list, p=prob_list)
            state.apply_action(action)
        else:
            if state.current_player() == 0:
                # player 1's turn
                action = bot1.step(state)
            else:
                # player 2's turn
                action = bot2.step(state)

            print("Player {} takes action {} at state {}".format(state.current_player(), action, state))

            state.apply_action(action)

    # Episode is over, update return
    returns = state.returns()
    print("Player 1: {}".format(returns[0]))
    print("Player 2: {}".format(returns[1]))
    return state

