from abc import ABC
from typing import List

class Dialogue(ABC):
    """
    The abstract class for dialogues.
    """
    def __init__(self):
        self.dialogue_list = []

    def dialogue_tuple_to_list(self) -> List[str]:
        """
        Convert dialogue tuples of (speaker, dialogue) to a list of dialogues of "speaker says: dialogue".
        """
        return self._dialogue_tuple_to_list()
    
    def get_list(self):
        return self.dialogue_list

    def _dialogue_tuple_to_list(self):
        pass

    def append(self, speaker: int, dialogue: str):
        """
        Append a dialogue to the dialogue list.
        """
        self.dialogue_list.append((speaker, dialogue))


class AvalonDiagloue(Dialogue):
    """
    The class for Avalon dialogue.
    """
    def __init__(self):
        super().__init__()

    def _dialogue_tuple_to_list(self) -> List[str]:
        dialogue_list = []
        for speaker, dialogue in self.dialogue_list:
            dialogue_list.append(f"Player {speaker} says: {dialogue}")

        return dialogue_list
