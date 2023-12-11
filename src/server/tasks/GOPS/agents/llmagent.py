import numpy as np
from typing import List
from .agent import GOPSAgent
from ..wrapper import SessionWrapper
from ..prompts import INITIALIZATION_MESSAGE
from ..gops_exception import GOPSAgentActionException
class LLMGOPSAgent(GOPSAgent):
    def __init__(self, id: int, hand: List[int], session: SessionWrapper, **kargs) -> None:
        super().__init__(
            id       =    id,
            hand     =    hand,
        )
        self.session = session
        self.your_score = 0
        self.opponent_score = 0
        self.history_stats = []

    def __repr__(self) -> str:
        return "Player {}".format(self.id)
    
    async def initialize(self) -> None:
        self.session.inject({
            "role": "user",
            "content": INITIALIZATION_MESSAGE.format(self.id, self.hand)
        })

    async def observe_round(self, contested_points: int, your_card: int, opponent_card: int, round_id: int) -> None:
        verbal_result = {
            True: "you win",
            False: "your opponent wins"
        }
        if your_card != opponent_card:
            await self.session.action({
                "role": "user",
                "content": f"Your played {your_card} this round. Your oppoenent played {opponent_card} this round. Thus {verbal_result[your_card > opponent_card]} {contested_points} points this round.",
                "mode": "system"
            })
        else:
            await self.session.action({
                "role": "user",
                "content": f"Your played {your_card} this round. Your oppoenent played {opponent_card} this round. Thus you and your opponent have a draw this round. The points will be added to the next round.",
                "mode": "system"
            })
        winner_verbal = ''
        if your_card > opponent_card:
            winner_verbal = f"You win {contested_points} scores this round"
            self.your_score += contested_points
        elif your_card < opponent_card:
            winner_verbal = f"Your opponent wins {contested_points} scores this round"
            self.opponent_score += contested_points
        elif your_card == opponent_card:
            winner_verbal = f"You and your opponent draw a tie this round. {contested_points} will be added to the next round"
        history_prompt = f"""Round ID: {round_id}
You played: {your_card}
Your opponent played: {opponent_card}
Result: {winner_verbal}

Total score for each player after this round:
You: {self.your_score}
Your opponent: {self.opponent_score}
"""
        self.history_stats.append(history_prompt)

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
    
    async def step(self, state: str, opponent_hand: List, contested_scores: int, score_card_left: List) -> int:
        print(f"This is player {self.id}, my hand is {self.hand}")
        history_stats = "\n".join(self.history_stats)
        game_prompt = f"""
Game Information:
The current score can be obtained by the winner of this round is {contested_scores}
The left score cards in the deck include {list(score_card_left)}

Your opponent's information:
Your opponent's hand is {list(opponent_hand)}
Your opponent's total score is {self.opponent_score}

Your information:
Your current hand is {list(self.hand)}.
Your current score is {self.your_score}

Please first consider information from both you and your opponent, the cards left in the deck, the score for this round, and whether if you need to win this round for your long-term goal, which is to win the game. And then make your decision to choose the card you want to play.
Please play a card from your hand: {self.hand}, and answer with the following format

Thought:
Your thought here

Decision:
The card you choose
"""
        card = -1
        cards = await self.session.action({
            "role": "user",
            "content": game_prompt,
            "mode": "play_card"
        })
        for possible_card in cards:
            possible_card = int(possible_card)
            if possible_card in self.hand:
                card = possible_card
                break
        if card not in self.hand:
            cards = await self.session.action({
                "role": "user",
                "content": f"You do not have that card in your hand. Please play a card from your hand {self.hand}.",
                "mode": "play_card"
            })
        for possible_card in cards:
            possible_card = int(possible_card)
            if possible_card in self.hand:
                card = possible_card
                break
        if card not in self.hand:
            raise GOPSAgentActionException("Invalid card with retry.")
        # card_index = self.hand.index(card)
        self.hand = np.delete(np.array(self.hand), np.where(np.array(self.hand) == card)).tolist()
        print("Current hand: ", self.hand)
        return int(card)