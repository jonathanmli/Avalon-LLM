from search_src.searchlight.headers import *
from search_src.searchlight.datastructures.adjusters import *
from search_src.searchlight.datastructures.beliefs import *
from search_src.searchlight.datastructures.estimators import *
from search_src.searchlight.classic_models import *
from search_src.GOPS.engine import *
from search_src.searchlight.datastructures.graphs import ValueGraph
import numpy as np
from tqdm import tqdm
from search_src.GOPS.baseline_models_GOPS import GOPSState2
from search_src.searchlight.gameplay.agents import SearchAgent

def compare_search(search1: Search, search2: Search, graph1: ValueGraph, graph2: ValueGraph, num_games: int, env):
    '''
    Evaluate two search algorithms by playing num_games games and comparing their performance.

    Args:
        search1: first search algorithm
        search2: second search algorithm
        num_games: number of games to play

    Returns:
        Player 1 win rate, player 2 win rate, player 1 average nodes expanded, player 2 average nodes expanded, player 1 average score, player 2 average score
    '''

    # TODO: make this generalizable for Avalon
    player_1_win_count = 0
    player_2_win_count = 0
    player_1_nodes_expanded = 0
    player_2_nodes_expanded = 0
    player_1_total_score = 0
    player_2_total_score = 0

    for i in tqdm(range(num_games)):
        (score_1, score_2, nodes_1, nodes_2) = play_gops_game(search1, search2, graph1, graph2, env)
        if score_1 > score_2:
            player_1_win_count += 1
        elif score_2 > score_1:
            player_2_win_count += 1
        player_1_nodes_expanded += nodes_1
        player_2_nodes_expanded += nodes_2
        player_1_total_score += score_1
        player_2_total_score += score_2
    print('player 1 rate', player_1_win_count/num_games)
    print('player 2 rate', player_2_win_count/num_games)
    print('player 1 average nodes expanded', player_1_nodes_expanded/num_games)
    print('player 2 average nodes expanded', player_2_nodes_expanded/num_games)
    print('player 1 average score', player_1_total_score/num_games)
    print('player 2 average score', player_2_total_score/num_games)
    return (player_1_win_count/num_games, player_2_win_count/num_games, player_1_nodes_expanded/num_games, player_2_nodes_expanded/num_games, player_1_total_score/num_games, player_2_total_score/num_games)

def play_gops_game(search1: Search, search2: Search, graph1: ValueGraph, graph2: ValueGraph, env: GOPSEnvironment, ):
    # reset search
    search1.reset()
    search2.reset()

    # play game
    player_1_played_cards = []
    player_2_played_cards = []
    prize_cards_played = []
    num_cards = env.get_num_turns()

    (done, score_card, contested_points) = env.reset()
    prize_cards_played.append(score_card)
    while not done:
        # print('score card', score_card)

        # player 1
        state = GOPSState2(
            {0,1},
            prize_cards=tuple(prize_cards_played),
            player_cards=tuple(player_1_played_cards),
            opponent_cards=tuple(player_2_played_cards),
            num_cards=num_cards
        )
        search1.expand(graph1, state)
        action_1 = search1.get_best_action(graph1, state, actor=0)

        # player 2
        state = GOPSState2(
            {1,0},
            prize_cards=tuple(prize_cards_played),
            player_cards=tuple(player_2_played_cards),
            opponent_cards=tuple(player_1_played_cards),
            num_cards=num_cards
        )
        search2.expand(graph2, state)
        action_2 = search2.get_best_action(graph2, state, actor=0)

        # print('nodes expanded minimax', self.search_2.get_nodes_expanded())

        # test if actions are the same
        # self.assertEqual(action_1, action_2)

        # print('player 1 action', action_1)
        # print('player 2 action', action_2)

        # update game state
        player_1_played_cards.append(action_1)
        player_2_played_cards.append(action_2)
        (done, score_card, contested_points) = env.play_cards(action_1, action_2)

        # update prize cards
        prize_cards_played.append(score_card)

    # print score
    # print('player 1 score', self.env.get_player1_score())
    # print('player 2 score', self.env.get_player2_score())

    # # print nodes expanded
    # print('player 1 nodes expanded', self.search_1.get_total_nodes_expanded())
    # print('player 2 nodes expanded', self.search_2.get_total_nodes_expanded())
        
    # return player scores and total nodes expanded
    return (env.get_player1_score(), env.get_player2_score(), search1.get_total_nodes_expanded(), search2.get_total_nodes_expanded())
            
def compare_search2(search1: Search, search2: Search, graph1: ValueGraph, graph2: ValueGraph, simulator: GameSimulator, num_games: int = 100, rng: np.random.Generator = np.random.default_rng()):
    '''
    Evaluate two search algorithms by playing num_games games and comparing their performance.

    Args:
        search1: first search algorithm
        search2: second search algorithm
        graph1: graph for search1
        graph2: graph for search2
        simulator: game simulator
        num_games: number of games to play

    Returns:
        Player 1 win rate, player 2 win rate, player 1 average nodes expanded, player 2 average nodes expanded, player 1 average score, player 2 average score
    '''

    # create agents
    agent1 = SearchAgent(search1, graph1, player=0)
    agent2 = SearchAgent(search2, graph2, player=1)
    agents = {0: agent1, 1: agent2, -1: RandomAgent(rng)}

    # play games
    avg_scores, trajectories = simulator.simulate_games(agents, num_games, display=True)

    # print results
    print('Player 1 win rate:', avg_scores[0])
    print('Player 2 win rate:', avg_scores[1])
    return avg_scores

    




