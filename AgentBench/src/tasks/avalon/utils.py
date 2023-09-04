from typing import Dict

def load_avalon_log(game_log: Dict):
    pass

def get_statement(last_history: str):
    return last_history.split("Statement: ")[-1]

if __name__ == "__main__":
    load_avalon_log()