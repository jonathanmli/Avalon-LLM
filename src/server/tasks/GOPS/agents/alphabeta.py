import numpy as np
from typing import List
from .agent import GOPSAgent
from ..wrapper import SessionWrapper
from open_spiel.python.algorithms import minimax

def construct_value_returns(id):
    def value_returns(state):
        return state.player_return(id)
    return value_returns

class AlphaBetaBot(GOPSAgent):
    """AlphaBetaBot implementation."""

    def __init__(self, id: int, hand: List[int], session: SessionWrapper, **kargs) -> None:
        super().__init__(
            id       =    id,
            hand     =    hand,
        )
        rng = np.random.RandomState()
        self.session = session
        self.rng = rng

    async def step(self, state, **kargs):
        """Returns the action to be taken by this bot in the given state."""
        legal_actions = state.legal_actions(self.id)
        value, action = minimax.expectiminimax(state, 5, construct_value_returns(self.id), self.id)
        print("Action from ABBot: ", action)
        return action