import numpy as np
from typing import List
from .agent import GOPSAgent
from ..wrapper import SessionWrapper
from ..prompts import INITIALIZATION_MESSAGE
from ..gops_exception import GOPSAgentActionException
class LLMGOPSAgent(GOPSAgent):
    def __init__(self, id: int, hand: List[int], session: SessionWrapper) -> None:
        super().__init__(
            id       =    id,
            hand     =    hand,
        )
        self.session = session

    def __repr__(self) -> str:
        return "Player {}".format(self.id)
    
    async def initialize(self) -> None:
        self.session.inject({
            "role": "user",
            "content": INITIALIZATION_MESSAGE.format(self.id, self.hand)
        })

    async def observe_round(self, contested_points: int, score_card: int, your_card: int, opponent_card: int) -> None:
        verbal_result = {
            True: "you",
            False: "your opponent"
        }
        await self.session.action({
            "role": "user",
            "content": f"Your played {your_card} last round. Your oppoenent played {opponent_card} last round. Thus {verbal_result[your_card > opponent_card]} won {contested_points} last round.",
            "mode": "system"
        })

    async def play_card(self, contested_points, score_card) -> int:
        print(f"This is player {self.id}, my hand is {self.hand}")
        card = await self.session.action({
            "role": "user",
            "content": f"Your current hand is {self.hand}.\nCurrent contested points: {contested_points}\nCurrent score card: {score_card}\nPlease play a card from your hand.",
            "mode": "play_card"
        })
        card = int(card)
        if card not in self.hand:
            card = await self.session.action({
                "role": "user",
                "content": "You do not have that card in your hand. Please play a card from your hand.",
                "mode": "play_card"
            })
            card = int(card)
        if card not in self.hand:
            raise GOPSAgentActionException("Invalid card with retry.")
        self.hand = np.delete(self.hand, np.where(self.hand == card))
        return int(card)