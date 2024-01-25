from Search.headers import *
from Search.search import *
from Search.beliefs import *
from Search.baseline_models_GOPS import *
from Search.estimators import *
from Search.classic_models import *
from GOPS.engine import *
import unittest
import numpy as np
from tqdm import tqdm

class ABMinimaxGOPSCompare(unittest.TestCase):

    # create random state
    random_state = np.random.RandomState(12)

    # create graph, one for each player
    graph_1 = ValueGraph()
    graph_2 = ValueGraph()

    # create config
    action_enumerator = GOPSActionEnumerator()
    action_predictor = GOPSRandomActionPredictor()
    forward_transitor = GOPSForwardTransitor()
    utility_estimator = UtilityEstimatorLast()
    actor_enumerator = GOPSActorEnumerator()
    value_heuristic_1 = RandomRolloutValueHeuristic(actor_enumerator, action_enumerator,
                                                    forward_transitor, num_rollouts=100, 
                                                    random_state=random_state)
    value_heuristic_2 = RandomRolloutValueHeuristic(actor_enumerator, action_enumerator,
                                                    forward_transitor, num_rollouts=100, 
                                                    random_state=random_state)

    # create search
    search_1 = SMAlphaBetaMinimax(forward_transitor, value_heuristic_1, 
                                  actor_enumerator, action_enumerator, 
                                    action_predictor, utility_estimator)
    search_2 = SMMinimax(forward_transitor, value_heuristic_2,
                            actor_enumerator, action_enumerator, 
                            action_predictor, utility_estimator)
    
    # create game
    num_cards = 6
    config = GOPSConfig(num_turns=num_cards, random_state=random_state)
    env = GOPSEnvironment(config)

    num_games = 100

    def test_compare(self):
        player_1_win_count = 0
        player_2_win_count = 0
        for i in tqdm(range(self.num_games)):
            (score_1, score_2) = self.play_game()
            if score_1 > score_2:
                player_1_win_count += 1
            elif score_2 > score_1:
                player_2_win_count += 1
        print('player 1 rate', player_1_win_count/self.num_games)
        print('player 2 rate', player_2_win_count/self.num_games)

    def play_game(self):
        # play game
        player_1_played_cards = []
        player_2_played_cards = []
        prize_cards_played = []

        (done, score_card, contested_points) = self.env.reset()
        prize_cards_played.append(score_card)
        while not done:
            # print('score card', score_card)

            # player 1
            state = GOPSState(
                {0,1},
                prize_cards=tuple(prize_cards_played),
                player_cards=tuple(player_1_played_cards),
                opponent_cards=tuple(player_2_played_cards),
                num_cards=self.num_cards
            )
            self.search_1.expand(self.graph_1, state, depth=3)
            action_1 = self.graph_1.get_best_action(state)

            # player 2
            state = GOPSState(
                {1,0},
                prize_cards=tuple(prize_cards_played),
                player_cards=tuple(player_2_played_cards),
                opponent_cards=tuple(player_1_played_cards),
                num_cards=self.num_cards
            )
            self.search_2.expand(self.graph_2, state, depth=3)
            action_2 = self.graph_2.get_best_action(state)

            # test if actions are the same
            # self.assertEqual(action_1, action_2)

            # print('player 1 action', action_1)
            # print('player 2 action', action_2)

            # update game state
            player_1_played_cards.append(action_1)
            player_2_played_cards.append(action_2)
            (done, score_card, contested_points) = self.env.play_cards(action_1, action_2)

            # update prize cards
            prize_cards_played.append(score_card)

        # print score
        # print('player 1 score', self.env.get_player1_score())
        # print('player 2 score', self.env.get_player2_score())
            
        # return player scores
        return (self.env.get_player1_score(), self.env.get_player2_score())
            



    




