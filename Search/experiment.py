from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatAnthropic
from Search.beliefs import ValueGraph
# from Tree_Search.headers import ValueBFS
from Search.headers import State
from Search.search import *
# from dynamics import *
from Search.baseline_models_GOPS import *
from Search.engine import *
from Search.estimators import *
import logging
from datetime import datetime
from tqdm import tqdm
from Search.classic_models import *

SYS_PROMPT = """You are a player in a GOPS (Game of pure strategy) game. The game has two players, and is played with a deck of cards. Each player is dealt a hand of cards. \
The goal of the game is to get the highest total scores. In each round, a player is asked to play a card from the hand to win the current score. The player who plays the highest card wins the round. \
The player who wins the most scores wins the game.\
Basically, you need to win the rounds with high scores. And you should also consider what cards left for you and your opponent to decide your strategy.\
"""

# Prepare logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# set output dir of logger
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'Search/output/output_log_{timestamp}.log'
fh = logging.FileHandler(filename, mode='w')
logger.addHandler(fh)

if __name__ == "__main__":
    class GPT35:
        def __init__(self):
            import os
            key = os.environ.get("OPENAI_API_KEY")
            
            self.model = ChatOpenAI(temperature=0.1, openai_api_key=key)
        def single_action(self, input_prompt: str):
            input_prompt = [HumanMessage(content=SYS_PROMPT), HumanMessage(content=input_prompt)]
            output = self.model(input_prompt).content
            # print(input_prompt)
            # print(output)

            return output
        
    class Claude:
        def __init__(self):
            import os
            key = os.environ.get("CLAUDE_API_KEY")

            self.model = ChatAnthropic(model="claude-2", temperature=0.1, anthropic_api_key=key)
        def single_action(self, input_prompt: str):
            input_prompt = [HumanMessage(content=SYS_PROMPT), HumanMessage(content=input_prompt)]
            output = self.model(input_prompt).content
            # print(input_prompt)
            logger.info(output)

            return output
        
    class RandomModel:
        def __init__(self) -> None:
            pass

        def single_action(self, input_prompt: str):
            pass
        
    class RandomPlayer:
        def __init__(self, cards: List[int]):
            self.hand = cards

        def single_action(self):
            import random
            card = random.choice(self.hand)
            self.hand.remove(card)
            return card

    model = GPT35()
    # model = RandomModel()

    # Instantiate the dynamics
    action_enumerator = GOPSActionEnumerator()
    # value_heuristic = GPT35ValueHeuristic(model)
    
    # opponent_action_predictor = GPT35OpponentActionPredictor(model)
    opponent_action_enumerator = GOPSOpponentActionEnumerator()
    hidden_state_predictor = GOPSRandomStatePredictor()
    hidden_state_enumerator = GOPSRandomStateEnumerator()
    forward_transitor = GOPSForwardTransitor()
    utility_estimator = UtilityEstimatorLast()
    value_heuristic = RandomRolloutValueHeuristic(action_enumerator, opponent_action_enumerator, 
                                                  forward_transitor, hidden_state_enumerator)
    
    num_cards = 6
    # using the engine defined earlier
    config = GOPSConfig(num_turns=num_cards)
    env = GOPSEnvironment(config)

    # bfs = ValueBFS(
    #     forward_transistor=forward_transitor,
    #     value_heuristic=value_heuristic, 
    #     action_enumerator=action_enumerator, 
    #     random_state_enumerator=hidden_state_enumerator,
    #     random_state_predictor=hidden_state_predictor,
    #     opponent_action_enumerator=opponent_action_enumerator,
    #     opponent_action_predictor=opponent_action_predictor,
    #     utility_estimator=utility_estimator
    # )

    bfs = SMMinimax(forward_transitor, value_heuristic, action_enumerator, 
                    hidden_state_enumerator, hidden_state_predictor,
                    opponent_action_enumerator,
                    utility_estimator)

    knowledge_graph = ValueGraph() # what the player knows about the game
    iter_num = 100
    player_wins = 0

    for iter in tqdm(range(iter_num)):
        logger.info(f"Iter: {iter}")
        played_prize_cards = [] # prize/score cards that have been shown to the players
        player_cards = []
        opponent_cards = []
        opponent = RandomPlayer(list(range(1, num_cards+1)))
        (done, score_card, contested_points) = env.reset()
        while not done:
            # Instantiate the search
            played_prize_cards.append(score_card)

            state = GOPSState(
                state_type='simultaneous',
                prize_cards=tuple(played_prize_cards),
                player_cards=tuple(player_cards),
                opponent_cards=tuple(opponent_cards),
                num_cards=num_cards
            )
            # print('Root state', state)
            bfs.expand(
                graph = knowledge_graph,
                state = state,
                depth = 3,
                render=False
            )

            player_card = knowledge_graph.get_best_action(state=state)
            opponent_card = opponent.single_action()

            player_cards.append(player_card)
            opponent_cards.append(opponent_card)

            # print("Played Prize Cards: {played_prize_cards}, Player plays {player_card}, Opponent plays {opponent_card}".format(
            #     played_prize_cards=played_prize_cards,
            #     player_card=player_card,
            #     opponent_card=opponent_card
            # ))

            # update the game state
            (done, score_card, contested_points) = env.play_cards(
                player1_card=player_card,
                player2_card=opponent_card
            )

        print("Player score: {player_score}, Opponent score: {opponent_score}".format(player_score=env.player1_score, opponent_score=env.player2_score))
        if env.player1_score > env.player2_score:
            player_wins += 1

    # print winrate of the player across all games
    print("Winrate: {}".format(player_wins/iter_num))


# run with python -m Search.test_search