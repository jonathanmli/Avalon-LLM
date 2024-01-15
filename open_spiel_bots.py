""""
Bots that work on OpenSpiel game environments
"""

# import libraries
import random
import pyspiel
import numpy as np
from open_spiel.python.algorithms import mcts, minimax, tabular_qlearner, nash_averaging
from Search.


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

    def __init__(self, env, player_id, rng=None):
        """Initializes the AlphaBetaBot."""
        super().__init__(env, player_id, rng)

    def step(self, state):
        """Returns the action to be taken by this bot in the given state."""
        legal_actions = state.legal_actions(self.player_id)
        value, action = minimax.expectiminimax(state, 5, self.construct_value_returns(self.player_id), self.player_id)
        return action
    
    def construct_value_returns(self, id):
        def value_returns(state):
            return state.player_return(id)
        return value_returns

class MCTSBot(OpenSpielBot):
    """MCTSBot implementation."""

    def __init__(self, env, player_id, rng=None, evaluator=None, uct_c=2, max_simulations=1000):
        """Initializes the MCTSBot."""
        super().__init__(env, player_id, rng)
        # Set up a simple evaluator: A uniform random bot used for rollouts if none is provided.
        if evaluator is None:
            evaluator = mcts.RandomRolloutEvaluator(random_state=self.rng)

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
    

    

    
