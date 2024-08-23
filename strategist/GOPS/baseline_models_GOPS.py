from collections.abc import Hashable
import os
import re
from typing import Dict, List, Union
from strategist.searchlight.headers import *
# from search_src.Search.prompts import *
import strategist.searchlight.utils as utils
from strategist.searchlightimprove.prompts.improvement_prompts import *

# import dataclasses
from dataclasses import dataclass

# Parse helper funcs
def parse_bracketed_list(string: str) -> List[str]:
    pattern = r'\[([^\]]+)\]'

    matches = re.findall(pattern, string)

    items = [item.strip() for item in matches[0].split(',')] if matches else []

    return items

def parse_dict_with_any_key(text):
    pattern = r'\{.*?\}'

    matches = re.findall(pattern, text)

    return matches[-1]

def parse_int_value(string: str) -> int:
    pattern = r'\b\d+\b'

    integers = [int(num) for num in re.findall(pattern, string)]

    return integers[-1] if len(integers) > 0 else None # should be designed in the prompt that the last num is the value

def parse_prob_value(string: str) -> float:
    pattern = r'\b\d+\.\d+|\b\d+|\.\d+\b'

    floats = [float(num) for num in re.findall(pattern, string)]

    return floats[-1] if len(floats) > 0 else None

@dataclass(frozen=True)
class GOPSState:
    '''
    GOPS state convention:

    ((1,2,5), (2,3), (4,1), 6,)

    (1,2,5 )are the prize cards shown to the player in that order
    (2,3) are the cards that player 0 has played in that order
    (4,1) are the cards that player 1 has played in that order
    6 is the total number of cards in each deck or hand
    '''
    player_0_played_cards: tuple
    player_1_played_cards: tuple
    played_prize_cards: tuple
    num_cards: int
    acting_players: tuple
    # simultaneous_actions: tuple[tuple[int, int], ...]

    def is_done(self) -> bool:
        '''
        Returns whether the game is done
        '''
        return len(self.player_0_played_cards) >= self.num_cards
    
    def is_almost_done(self) -> bool:
        '''
        Returns whether the game is almost done
        '''
        return len(self.played_prize_cards) >= self.num_cards
    
    def get_reward_player_1(self) -> int:
        '''
        Returns the reward of the state
        '''
        scores = self.calculate_score()
        return scores[1] - scores[0]
    
    def get_reward_player_0(self) -> int:
        '''
        Returns the reward of the state
        '''
        scores = self.calculate_score()
        return scores[0] - scores[1]
    
    def get_prize_deck(self) -> set:
        '''
        Returns the prize deck
        '''
        return set(range(1, self.num_cards+1)) - set(self.played_prize_cards)

    def get_player_0_hand(self) -> set:
        '''
        Returns player 0's hand
        '''
        return set(range(1, self.num_cards+1)) - set(self.player_0_played_cards)

    def get_player_1_hand(self) -> set:
        '''
        Returns player 1's hand
        '''
        return set(range(1, self.num_cards+1)) - set(self.player_1_played_cards)
    
    def get_acting_player(self) -> int:
        '''
        Returns the acting player
        '''
        return self.acting_players[0]

    def calculate_score(self) -> dict[int, int]:
        '''
        Calculates the score of the state for both players
        '''
        contested_points = 0
        player_0_score = 0
        player_1_score = 0
        for idx, single_score in enumerate(list(self.played_prize_cards)):
            contested_points += single_score
            if idx >= len(self.player_0_played_cards) or idx >= len(self.player_1_played_cards):
                break
            if self.player_0_played_cards[idx] > self.player_1_played_cards[idx]:
                player_0_score += contested_points
                contested_points = 0
            elif self.player_0_played_cards[idx] < self.player_1_played_cards[idx]:
                player_1_score += contested_points
                contested_points = 0
        return {0: player_0_score, 1: player_1_score}
    
    @staticmethod
    def init_from_num_cards(num_cards: int) -> 'GOPSState':
        '''
        Returns the initial state of the game
        '''
        return GOPSState((), (), (), num_cards, (-1,),)
        
class GOPSForwardTransitor(ForwardTransitor):

    def __init__(self, default_player_order: tuple[int, ...] = (0, 1)):
        super().__init__()
        self.default_player_order = default_player_order

    def _transition(self, state: GOPSState, action: int) ->tuple[GOPSState, dict[int, float]]:
        reward = {0: 0.0, 1: 0.0}

        # there are generally three cases
        if not state.acting_players: # terminal state
            raise ValueError('Terminal state, no actions should be taken')
        elif -1 in state.acting_players: # random state
            # make sure the action is in the prize deck
            assert action in state.get_prize_deck()
            # append the action to the prize cards
            played_prize_cards = list(state.played_prize_cards)
            played_prize_cards.append(action)
            played_prize_cards = tuple(played_prize_cards)
            # change the acting players to self.default_player_order
            acting_players = self.default_player_order
            return GOPSState(state.player_0_played_cards, state.player_1_played_cards, played_prize_cards, state.num_cards, acting_players,), reward
        elif 0 in state.acting_players or 1 in state.acting_players: # simultaneous state 
            acting_player = state.get_acting_player()
            # make sure the action is in the acting player's hand
            assert action in state.get_player_0_hand() if acting_player == 0 else action in state.get_player_1_hand()
            # append the action to the acting player's played cards
            player_0_played_cards = state.player_0_played_cards
            player_1_played_cards = state.player_1_played_cards
            if acting_player == 0:
                player_0_played_cards = list(player_0_played_cards)
                player_0_played_cards.append(action)
                player_0_played_cards = tuple(player_0_played_cards)
            else:
                player_1_played_cards = list(player_1_played_cards)
                player_1_played_cards.append(action)
                player_1_played_cards = tuple(player_1_played_cards)
            # remove the acting player from the acting players
            acting_players = tuple(state.acting_players[1:])
            
            # see if acting_players is empty
            if not acting_players:
                if state.is_almost_done():
                    acting_players = tuple()
                else:
                    acting_players = (-1,)

            new_state = GOPSState(player_0_played_cards, player_1_played_cards, state.played_prize_cards, state.num_cards, acting_players,)

            # now calculate the reward
            if not acting_players:
                scores = new_state.calculate_score()
                player_0_score = scores[0]
                player_1_score = scores[1]
                reward[0] = player_0_score - player_1_score
                reward[1] = player_1_score - player_0_score

            return new_state, reward
        else:
            raise ValueError('Invalid actors'+str(state.acting_players))

class GOPSActorActionEnumerator(ActorActionEnumerator):

    def __init__(self):
        super().__init__()

    def _enumerate(self, state: GOPSState) -> tuple[Union[int, None], set]:
        if not state.acting_players:
            return None, set()
        elif state.get_acting_player() == 0:
            return 0, set(state.get_player_0_hand())
        elif state.get_acting_player() == 1:
            return 1, set(state.get_player_1_hand())
        elif state.get_acting_player() == -1:
            return -1, set(state.get_prize_deck())
        else:
            raise ValueError('Invalid actors'+str(state.acting_players))

    @staticmethod
    def parse_str_to_action(action_str: str) -> int:
        return int(action_str)

class GOPSFunctionalValueHeuristic(ValueHeuristic):
    '''
    Functional value heuristic for LLMs
    '''

    EVAL_TEST_STATES = [
        ((1, 3, 2, 6, 5), (2, 3, 4, 1, 6), (1, 3, 4, 6, 5), False, 6, 11, {4,}, {5,}, {2,}),
        ((1, 3, 2), (2, 3, 4), (1, 3, 2), False, 6, 0, {4, 5, 6}, {1, 5, 6}, {2, 4, 6}),
        ((1, 3, 2, 4), (2, 3, 1, 4), (1, 3, 2, 6), False, 1, 9, {5, 6}, {5, 6}, {4, 5}),
        ((1, 3), (2, 3), (1, 2), False, 4, 0, {2, 4, 5, 6}, {1, 4, 5, 6}, {3, 4, 5, 6}),
        ((1, 3, 6), (2, 1, 4), (1, 3, 5), False, 1, 9, {2, 4, 5}, {3, 5, 6}, {2, 4, 6}),
        ((1,), (1,), (6,), False, 0, 1, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {1, 2, 3, 4, 5}),
        ((1, 4), (1, 3), (2, 5), False, 0, 5, {2, 3, 5, 6}, {2, 4, 5, 6}, {1, 3, 4, 6}),
        ((4, 3, 1, 2, 6), (2, 3, 4, 1, 6), (2, 3, 4, 5, 1), False, 6, 10, {5,}, {5,}, {6,}),
        ((4, 3), (2, 1), (2, 4), False, 0, 7, {1, 2, 5, 6}, {3, 4, 5, 6}, {1, 3, 5, 6}),
        ((4,), (2,), (5,), False, 0, 4, {1, 2, 3, 5, 6}, {1, 3, 4, 5, 6}, {1, 2, 3, 4, 6}),
        ((4, 3, 1, 2, 5), (2, 3, 4, 1, 6), (2, 3, 4, 6, 5), False, 5, 10, {6,}, {5,}, {1,}),
        ((4,), (3,), (2,), False, 4, 0, {1, 2, 3, 5, 6}, {1, 2, 4, 5, 6}, {1, 3, 4, 5, 6}),
        ((4, 2, 6, 5, 1, 3), (6, 5, 4, 3, 2), (6, 5, 4, 3, 2), True, 0, 0, set(), {1}, {1}),
        ((3, 1, 2), (3, 2, 1), (3, 2, 1), False, 0, 0, set(), set(), set()), # NOTE: this is an end state
        ((1, 2), (1,), (1,), True, 0, 0, {3}, {2, 3}, {2, 3}),
        ((2, 1, 3), (2, 1), (1, 3), True, 2, 1, set(), {3}, {2}),
        ((5,), (5,), (5,), False, 0, 0, {1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}),
        ((2,), (3,), (2,), False, 2, 0, {1, 3, 4, 5}, {1, 2, 4, 5}, {1, 3, 4, 5}),
    ]

    def __init__(self, func: str):
        '''
        Args:
            func: the function string
        '''
        self.test_evaluate(func)
        self.attach_func(func)

    def attach_func(self, function_str, safe_mode = False) -> bool:
        '''
        Attach the evaluate function
        '''
        if safe_mode:
            # Test the evaluate function
            exec(function_str, globals(), locals())
            
            # Attach the dynamically defined function to the instance
            self._llm_evaluate = locals()['evaluate_state']

        else:
            # Execute the function definition within the local scope of __init__
            exec(function_str, globals(), locals())
            
            # Attach the dynamically defined function to the instance
            self._llm_evaluate = locals()['evaluate_state']


    @staticmethod
    def test_evaluate(function_str, safe_mode = False) -> bool:
        '''
        Test the evaluate function
        '''
        if safe_mode:
            # Test the evaluate function
            try:
                exec(function_str, globals(), locals())
                
                # Attach the dynamically defined function to the instance
                llm_evaluate = locals()['evaluate_state']

                for state in GOPSFunctionalValueHeuristic.EVAL_TEST_STATES:
                    (player_value, opponent_value), notes = llm_evaluate(state)
                    # assert that both values are numbers
                    assert isinstance(player_value, (int, float))
                    assert isinstance(opponent_value, (int, float))
                    # assert that notes is a dictionary
                    assert isinstance(notes, dict)

                # print('successfully passed test')
            except Exception as e:  # Capture the exception as 'e'
                print(f"An exception occurred: {e}")  # Print the exception for debugging
                return False
            return True
        else:
            # Execute the function definition within the local scope of __init__
            exec(function_str, globals(), locals())
            
            # Attach the dynamically defined function to the instance
            llm_evaluate = locals()['evaluate_state']

            for state in GOPSFunctionalValueHeuristic.EVAL_TEST_STATES:
                (player_value, opponent_value), notes = llm_evaluate(state)
                # assert that both values are numbers
                assert isinstance(player_value, (int, float))
                assert isinstance(opponent_value, (int, float))
                # assert that notes is a dictionary
                assert isinstance(notes, dict)

            return True
        
    @staticmethod
    def parse_llm_function(function_str: str, safe_mode = False) -> str:
        # Parse the function definition
        pattern = re.compile(r'(?ms)^def\s+evaluate_state\s*\(.*?\)\s*.*?:\s*.*?(?=^\w|\Z)', re.MULTILINE)
        match = re.search(pattern, function_str)
        if match:
            # print(f"Parsed Function:\n{match.group()}")
            # logging.info(match.group()) #TODO: fix logging to match the new logging framework
            parsed_func = match.group()
            function_lines = parsed_func.split('\n')
            last_line_id = -2
            for line_id in range(len(function_lines)-1, -1, -1):
                if function_lines[line_id].lstrip().startswith("return"):
                    last_line_id = line_id
                    break
            return '\n'.join(function_lines[:last_line_id+1])
        else:
            if not safe_mode:
                raise ValueError("Parsing error: Function not found")
            return None
        

    def _evaluate(self, state: GOPSState) -> tuple[dict, dict]:
        # Prepare input
        player_0_played_cards = state.player_0_played_cards
        player_1_played_cards = state.player_1_played_cards
        prize_cards = state.played_prize_cards
        scores = state.calculate_score()
        is_player_turn = not -1 in state.acting_players
        remain_prize_cards = state.get_prize_deck()
        player_0_hand = state.get_player_0_hand()
        player_1_hand = state.get_player_1_hand()


        # # make sure the inputs are of the correct type
        # assert isinstance(player_cards, tuple)
        # assert isinstance(opponent_cards, tuple)
        # assert isinstance(prize_cards, tuple)
        # # assert isinstance(player_score, int)
        # # assert isinstance(opponent_score, int)
        # assert isinstance(is_player_turn, bool)
        # assert isinstance(remain_prize_cards, set)
        # assert isinstance(player_hand, set)
        # assert isinstance(opponent_hand, set)

        # use the function to calculate the value
        try:
            state_tup = (prize_cards, player_0_played_cards, player_1_played_cards, is_player_turn, scores[0], scores[1], remain_prize_cards, player_0_hand, player_1_hand)
            (player_value, opponent_value), notes = self._llm_evaluate(state_tup)
        # raise an error if the function is not defined properly
        except Exception as e:
            logging.warning(f"state tuple: {str(state_tup)}")
            # print('state tuple', state_tup)
            # NOTE: add any printed tuples to EVAL_TEST_STATES for future testing
            raise ValueError(f"Function not defined properly: {e}")
        
        return {0:player_value - opponent_value, 1:opponent_value - player_value}, notes

@dataclass(frozen=True)
class GOPSInformationSet(GOPSState):
    '''
    We assume cards in deck and hand are list(range(1, num_cards+1))
    '''
    # player_0_played_cards: tuple
    # player_1_played_cards: tuple
    # played_prize_cards: tuple
    # num_cards: int
    # acting_players: tuple

    def __str__(self) -> str:
        state_description = f'''The current state of the game is as follows:
        - The prize cards that have been revealed are: {self.played_prize_cards}
        - The cards that player 0 has played are: {self.player_0_played_cards}
        - The cards that player 1 has played are: {self.player_1_played_cards}
        - Player 0's score so far is: {self.calculate_score()[0]}
        - Player 1's score so far is: {self.calculate_score()[1]}
        - The prize cards left in the prize deck are: {self.get_prize_deck()}
        - The cards left in player 0's hand are: {self.get_player_0_hand()}
        - The cards left in player 1's hand are: {self.get_player_1_hand()}
        '''
        return state_description
    
class GOPSInformationFunction(InformationFunction):

    def _get_information_set(self, state: GOPSState, actor: int) -> GOPSInformationSet:
        player_0_played_cards = state.player_0_played_cards
        player_1_played_cards = state.player_1_played_cards

        # simply remove the last element from player_0_played_cards or player_1_played_cards, whichever one has one more element
        if len(player_0_played_cards) > len(player_1_played_cards):
            player_0_played_cards = player_0_played_cards[:-1]
        elif len(player_1_played_cards) > len(player_0_played_cards):
            player_1_played_cards = player_1_played_cards[:-1]

        return GOPSInformationSet(player_0_played_cards, player_1_played_cards, state.played_prize_cards, state.num_cards, state.acting_players)
    
    

class GOPSInformationPrior(InformationPrior):

    def _get_prior_state(self, information_set: GOPSInformationSet) -> GOPSState:
        return GOPSState(information_set.player_0_played_cards, information_set.player_1_played_cards, information_set.played_prize_cards, information_set.num_cards, information_set.acting_players,)
        # if len(information_set.player_0_played_cards) == len(information_set.played_prize_cards):
        #     return GOPSState(information_set.player_0_played_cards, information_set.player_1_played_cards, information_set.played_prize_cards, information_set.num_cards, (-1,),)
        # else:
        #     return GOPSState(information_set.player_0_played_cards, information_set.player_1_played_cards, information_set.played_prize_cards, information_set.num_cards, (0, 1),)