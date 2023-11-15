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

    def play_card(self, hand: List[int]) -> int:
        r"""Play a card from the hand.

        Args:
            hand (List[int]): The list of cards in the hand.

        Returns:
            int: The index of the card to play.
        """
        raise NotImplementedError