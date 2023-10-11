from typing import List

def vote(approval: bool):
    if approval:
        return 1
    else:
        return 0
    
def choose(player_list: List[int]):
    return player_list

def assassinate(player_id: int):
    return player_id