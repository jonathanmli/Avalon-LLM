import numpy as np
from typing import List
from .agent import GOPSAgent
from ..wrapper import SessionWrapper
# from open_spiel.python.algorithms import minimax
from open_spiel.python.algorithms import mcts

def construct_value_returns(id):
    def value_returns(state):
        return state.player_return(id)
    return value_returns

class MCTSBot(GOPSAgent):
    """MCTSBot implementation."""

    def __init__(self, id: int, hand: List[int], session: SessionWrapper, **kargs) -> None:
        super().__init__(
            id       =    id,
            hand     =    hand,
        )
        game = kargs.pop("game", None)
        assert game is not None

        # Create random state
        random_state = np.random.RandomState(12)

        # Set up a simple evaluator: A uniform random bot used for rollouts
        evaluator = mcts.RandomRolloutEvaluator(random_state=random_state)


        # Set up MCTS parameters
        self.mcts_bot = mcts.MCTSBot(
            game,
            uct_c=2,  # Exploration constant
            max_simulations=1000,  # Number of MCTS simulations per move
            evaluator=evaluator,  # Evaluator (rollout policy)
            random_state=random_state  # Random seed
        )
        rng = np.random.RandomState()
        self.session = session
        self.rng = rng

    async def step(self, state, **kargs):
        """Returns the action to be taken by this bot in the given state."""
        action = self.mcts_bot.step(state)
        print("Action from ABBot: ", action)
        return action