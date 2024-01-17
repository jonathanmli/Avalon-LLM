""""
Bots that work on OpenSpiel game environments
"""

# import libraries
import re
import random
import pyspiel
import numpy as np
from typing import List
from open_spiel.python.algorithms import mcts, minimax, tabular_qlearner, nash_averaging
from Search.beliefs import ValueGraph
from Search.headers import State
from Search.search import *
from Search.baseline_models_GOPS import *
from Search.engine import *
from Search.estimators import *
from Search.classic_models import *


def get_card_sequence(state: str) -> List[int]:
    target_string = "Point card sequence:"
    matches = re.search(f"{re.escape(target_string)}\s*([\d\s]+)", str(state))
    numbers = [int(num) for num in matches.group(1).split()]

    return numbers

def get_player1_hands(state: str) -> List[int]:
    target_string = "P0 hand:"
    matches = re.search(f"{re.escape(target_string)}\s*([\d\s]+)", str(state))
    numbers = [int(num) for num in matches.group(1).split()]

    return numbers

def get_player2_hands(state: str) -> List[int]:
    target_string = "P1 hand:"
    matches = re.search(f"{re.escape(target_string)}\s*([\d\s]+)", str(state))
    numbers = [int(num) for num in matches.group(1).split()]

    return numbers

def get_score_card(state: str) -> List[int]:
    target_string = "Point card sequence:"
    matches = re.search(f"{re.escape(target_string)}\s*([\d\s]+)", str(state))
    numbers = [int(num) for num in matches.group(1).split()]

    return numbers

def get_points(state: str) -> List[int]:
    target_string = "Points:"
    matches = re.search(f"{re.escape(target_string)}\s*([\d\s]+)", str(state))
    numbers = [int(num) for num in matches.group(1).split()]

    return numbers

def open_spiel_state_to_gops_state(open_spiel_state: str):
    """
    Converts an OpenSpiel state to a GOPS state
    """
    cards = get_card_sequence(open_spiel_state)
    player1_hands = get_player1_hands(open_spiel_state)
    player2_hands = get_player2_hands(open_spiel_state)
    score_card = get_score_card(open_spiel_state)
    points = get_points(open_spiel_state)
    contested_scores = sum(score_card) - sum(points)

    return GOPSState(
        state_type="simultaneous",
        prize_cards=cards,
        player_cards=player1_hands,
        opponent_cards=player2_hands,
        num_cards=len(cards),
    )
    

class OpenSpielBot:

    def __init__(self, env, player_id, rng=None) -> None:
        self.player_id = player_id
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng
        self.env = env

    def step(self, state):
        raise NotImplementedError


class RandomBot(OpenSpielBot):
    """Random bot implementation."""

    def __init__(self, env, player_id, rng=None):
        """Initializes the random bot."""
        super().__init__(env, player_id, rng)
      
    def step(self, state):
        """Returns the action to be taken by this bot in the given state."""
        legal_actions = state.legal_actions(self.player_id)
        return self.rng.choice(legal_actions)
    
    def __str__(self):  # pragma: no cover
        return "RandomBot"
    
    def __repr__(self):  # pragma: no cover
        return "RandomBot"
    
class AlphaBetaBot(OpenSpielBot):
    """AlphaBetaBot implementation."""

    def __init__(self, env, player_id, rng=None, evaluator=None, depth=5):
        """Initializes the AlphaBetaBot."""
        super().__init__(env, player_id, rng)
        self.depth = depth
        # Set up a simple evaluator: A uniform random bot used for rollouts if none is provided.
        if evaluator is None:
            evaluator = mcts.RandomRolloutEvaluator(n_rollouts=100, random_state=self.rng)
        self.evaluator = evaluator

    def step(self, state):
        """Returns the action to be taken by this bot in the given state."""
        legal_actions = state.legal_actions(self.player_id)
        value, action = minimax.expectiminimax(state, self.depth, self.construct_value_returns(), self.player_id)
        return action
    
    def construct_value_returns(self, id):
        def value_returns(state):
            return state.player_return(id)
        return value_returns
    
    def construct_value_returns(self):
        def value_returns(state):
            return self.evaluator.evaluate(state)[self.player_id]
        return value_returns

class MCTSBot(OpenSpielBot):
    """MCTSBot implementation."""

    def __init__(self, env, player_id, rng=None, evaluator=None, uct_c=2, max_simulations=100):
        """Initializes the MCTSBot."""
        super().__init__(env, player_id, rng)
        # Set up a simple evaluator: A uniform random bot used for rollouts if none is provided.
        if evaluator is None:
            evaluator = mcts.RandomRolloutEvaluator(n_rollouts=100, random_state=self.rng)

        # wrapper around the MCTS bot
        self.mcts_bot = mcts.MCTSBot(
            game=self.env,
            uct_c=uct_c,  # Exploration constant
            max_simulations=max_simulations,  # Number of MCTS simulations per move
            evaluator=evaluator,  # Evaluator (rollout policy)
            random_state=self.rng  # Random seed
        )

    def step(self, state):
        """Returns the action to be taken by this bot in the given state."""
        return self.mcts_bot.step(state)
    

    

    
class CustomBot:

    def __init__(self, player_id, rng=None) -> None:
        self.player_id = player_id
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng

    def step(self, state):
        raise NotImplementedError

class SMMinimaxCustomBot(CustomBot):

    def __init__(self, player_id, rng=None, max_depth=3, num_rollouts=100):
        """Initializes the SMMinimaxBot.
        
        Args:
            player_id: player id
            rng: random number generator
            max_depth: maximum depth to search
            num_rollouts: number of rollouts to perform for value estimation
        """
        super().__init__(player_id, rng)
        self.max_depth = max_depth
        self.value_graph = ValueGraph()
        self.action_enumerator = GOPSActionEnumerator()
        self.opponent_action_enumerator = GOPSOpponentActionEnumerator()
        self.hidden_state_enumerator = GOPSRandomStateEnumerator()
        self.hidden_state_predictor = GOPSRandomStatePredictor()
        self.forward_transitor = GOPSForwardTransitor()
        self.utility_estimator = UtilityEstimatorLast()
        self.value_heuristic = RandomRolloutValueHeuristic(self.action_enumerator, self.opponent_action_enumerator, 
                                                      self.forward_transitor, self.hidden_state_enumerator, 
                                                      num_rollouts=num_rollouts)
        self.search = SMMinimax(self.forward_transitor, self.value_heuristic, self.action_enumerator,
                                self.hidden_state_enumerator, self.hidden_state_predictor,
                                self.opponent_action_enumerator, self.utility_estimator)

    def step(self, state: GOPSState):
        """Returns the action to be taken by this bot in the given state."""
    
        # then expand the value graph
        self.search.expand(self.value_graph, state, depth=self.max_depth)

        # then get the best action from the value graph
        action = self.value_graph.get_best_action(state)

        return action - 1 # subtract 1 because OpenSpiel actions are 0-indexed
