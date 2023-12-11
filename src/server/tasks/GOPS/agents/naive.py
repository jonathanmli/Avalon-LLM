import numpy as np
from typing import List
from .agent import GOPSAgent
from ..wrapper import SessionWrapper
class NaiveGOPSAgent(GOPSAgent):
    def __init__(self, id: int, hand: List[int], session: SessionWrapper, **kargs) -> None:
        super().__init__(
            id       =    id,
            hand     =    hand,
        )
        self.session = session

    def __repr__(self) -> str:
        return "Player {}".format(self.id)

    async def play_card(self, contested_points: int, score_card: int) -> int:
        card_id = np.random.choice(len(self.hand))
        card = self.hand[card_id]
        self.hand = np.delete(self.hand, np.where(self.hand == card))
        return card
    
    async def step(self, state: str, opponent_hand: List, contested_scores: int, score_card_left: List) -> int:
        card_id = np.random.choice(len(self.hand))
        card = self.hand[card_id]
        self.hand = np.delete(np.array(self.hand), np.where(np.array(self.hand) == card)).tolist()
        return int(card)