import re
from typing import List

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