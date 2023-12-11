from typing import List
class GOPSAgent:
    r"""The base class for all agents.

    Args:
        id (int): The Player id of the agent.
        config (AvalonBasicConfig): The config of the agent.

    To implement your own agent, subclass this class and implement the following methods:
        - :method:`Agent.play_card`
    """
    def __init__(self, id: int, hand: List[int]) -> None:
        self.id = id
        self.hand = hand

    async def initialize(self) -> None:
        r"""Initialize the agent. This method is called once at the beginning of the game.

        Returns:
            None
        """
        pass

    async def observe_round(self, contested_points: int, your_card: int, opponent_card: int, round_id: int) -> None:
        r"""Observe the current round. This method is called at the beginning of each round.

        Args:
            contested_points (int): The number of contested points.
            score_card (int): The score card.
            your_card (int): The card you played last round.
            opponent_card (int): The card your opponent played last round.

        Returns:
            None
        """
        pass

    def play_card(self, hand: List[int]) -> int:
        r"""Play a card from the hand.

        Args:
            hand (List[int]): The list of cards in the hand.

        Returns:
            int: The index of the card to play.
        """
        raise NotImplementedError