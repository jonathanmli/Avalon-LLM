from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from Search.beliefs import ValueGraph
# from Tree_Search.headers import ValueBFS
from Search.headers import State
from Search.search import *
# from dynamics import *
from Search.baseline_models_GOPS import *

SYS_PROMPT = """You are a player in a GOPS (Game of pure strategy) game. The game has two players, and is played with a deck of cards. Each player is dealt a hand of cards. \
The goal of the game is to get the highest total scores. In each round, a player is asked to play a card from the hand to win the current score. The player who plays the highest card wins the round. \
The player who wins the most scores wins the game.\
Basically, you need to win the rounds with high scores. And you should also consider what cards left for you and your opponent to decide your strategy.\

In the current game, you are player 1. You have been dealt the following hand: [1,2,3,4,5,6]."""

if __name__ == "__main__":
    class GPT35:
        def __init__(self):
            import os
            key = os.environ.get("OPENAI_API_KEY")
            
            self.model = ChatOpenAI(temperature=0.1, openai_api_key=key)
        def single_action(self, input_prompt: str):
            input_prompt = [HumanMessage(content=SYS_PROMPT), HumanMessage(content=input_prompt)]
            output = self.model(input_prompt).content

            print(output)

            return output
        
    class RandomPlayer:
        def __init__(self, cards: List[int]):
            self.hand = cards

        def single_action(self):
            import random
            card = random.choice(self.hand)
            self.hand.remove(card)
            return card
        

    class GOPSSystem:
        def __init__(self, cards: List[int]):
            self.card_deck = cards

        def draw_score_card(self):
            import random
            card = random.choice(self.card_deck)
            self.card_deck.remove(card)
            return card

    model = GPT35()

    # Instantiate the game system
    system = GOPSSystem([1,2,3,4,5,6])

    opponent = RandomPlayer([1,2,3,4,5,6])

    # Instantiate the dynamics
    action_enumerator = GOPSActionEnumerator()
    value_heuristic = GPT35ValueHeuristic(model)
    opponent_action_predictor = GPT35OpponentActionPredictor(model)
    opponent_action_enumerator = GOPSOpponentActionEnumerator()
    hidden_state_predictor = GOPSRandomStatePredictor()
    hidden_state_enumerator = GOPSRandomStateEnumerator()
    forward_predictor = GOPSForwardPredictor()
    forward_enumerator = GOPSForwardEnumerator()

    player_hand = [1,2,3,4,5,6]
    player_cards = [] # cards that have been played
    opponent_cards = [] # cards that the opponent has played
    played_prize_cards = [] # prize/score cards that have been shwon to the players

    player_score = 0
    opponent_score = 0
    current_score = 0

    while len(system.card_deck) > 0:
        # Instantiate the search
        # TODO: do we need to instantiate the search every time?
        graph = ValueGraph()
        bfs = ValueBFS(
            forward_predictor=forward_predictor, 
            forward_enumerator=forward_enumerator, 
            value_heuristic=value_heuristic, 
            action_enumerator=action_enumerator, 
            random_state_enumerator=hidden_state_enumerator,
            random_state_predictor=hidden_state_predictor,
            opponent_action_enumerator=opponent_action_enumerator,
            opponent_action_predictor=opponent_action_predictor,
        )
        current_score_card = system.draw_score_card()
        current_score += current_score_card
        played_prize_cards.append(current_score_card)
        state = GOPSState(
            state_type=0,
            prize_cards=tuple(played_prize_cards),
            player_cards=tuple(player_cards),
            opponent_cards=tuple(opponent_cards),
            num_cards=6
        )
        # print("State: {state}".format(state=state))
        bfs.expand(
            graph = graph,
            state = state,
            depth = 3,
            render=True
        )
        player_card = graph.get_best_action(state=state)
        opponent_card = opponent.single_action()

        print("Played Prize Cards: {played_prize_cards}, Player plays {player_card}, Opponent plays {opponent_card}".format(
            played_prize_cards=played_prize_cards,
            player_card=player_card,
            opponent_card=opponent_card
        ))

        player_cards.append(player_card)
        opponent_cards.append(opponent_card)

        if player_card > opponent_card:
            player_score += current_score
            current_score = 0
        elif player_card < opponent_card:
            opponent_score += current_score
            current_score = 0

    print("Player score: {player_score}, Opponent score: {opponent_score}".format(player_score=player_score, opponent_score=opponent_score))

# run with python -m Search.test_search