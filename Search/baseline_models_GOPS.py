import re
from typing import Dict, List
from Search.headers import *
from Search.prompts import *

# Parse helper funcs
def parse_bracketed_list(string: str) -> List[str]:
    pattern = r'\[([^\]]+)\]'

    matches = re.findall(pattern, string)

    items = [item.strip() for item in matches[0].split(',')] if matches else []

    return items

def parse_dict_with_any_key(text):
    pattern = r'([^:{}]+)\s*:\s*(\w+)'

    matches = re.findall(pattern, text)

    return {key.strip(): value for key, value in matches}

def parse_int_value(string: str) -> int:
    pattern = r'\b\d+\b'

    integers = [int(num) for num in re.findall(pattern, string)]

    return integers[-1] if len(integers) > 0 else None # should be designed in the prompt that the last num is the value

def parse_prob_value(string: str) -> float:
    pattern = r'\b\d+\.\d+|\b\d+|\.\d+\b'

    floats = [float(num) for num in re.findall(pattern, string)]

    return floats[-1] if len(floats) > 0 else None


class GOPSState(State):
    '''
    GOPS state convention:

    ((1,2,5), (2,3), (4,1), 6, simultaneous)

    1,2,5 are the prize cards shown to the player in that order
    2,3 are the cards that the player has played in that order
    4,1 are the cards that the opponent has played in that order
    6 is the total number of cards in each deck or hand
    simultaneous is the state type, which can be one of the following:
        stochastic: a random card is revealed
        simultaneous: both players choose a card to play
    '''

    def __init__(self, state_type, prize_cards, player_cards, opponent_cards, num_cards):
        # should call super first, otherwise state_type will be overwritten
        id = tuple([prize_cards, player_cards, opponent_cards, num_cards, state_type])
        # turn = self.STATE_TYPES[state_type]
        super().__init__(id, state_type)

        self.prize_cards = tuple(prize_cards)
        self.player_cards = tuple(player_cards)
        self.opponent_cards = tuple(opponent_cards)
        self.num_cards = num_cards
        
class GOPSForwardTransitor(ForwardTransitor):

    def __init__(self):
        super().__init__()

    def transition(self, state: GOPSState, actions):
        '''
        Transitions to the next state given the current state and action

        Args:
            state: current state
            actions: actions taken by the protagonist and the antagonist (ie cards played by the player and the opponent), or prize card played by the environment

        Returns:
            next_state: next state
        '''

        # we need to be careful to copy the state, otherwise the state will be changed
        prize_cards = state.prize_cards
        player_cards = state.player_cards
        opponent_cards = state.opponent_cards
        num_cards = state.num_cards
        state_type = state.state_type

        if state_type == 'simultaneous': # simultaneous state
            # assert that len of actions is 2
            assert len(actions) == 2

            # assert that actions are not in the player_cards and between 1 and num_cards
            assert actions[0] not in player_cards and actions[0] in range(1, num_cards+1)

            # assert that actions are not in the opponent_cards and between 1 and num_cards
            assert actions[1] not in opponent_cards and actions[1] in range(1, num_cards+1)

            # append actions to player_cards and opponent_cards
            player_cards = list(player_cards)
            player_cards.append(actions[0])
            player_cards = tuple(player_cards)

            opponent_cards = list(opponent_cards)
            opponent_cards.append(actions[1])
            opponent_cards = tuple(opponent_cards)

            # change state_type to stochastic
            state_type = 'stochastic'

        elif state_type == 'stochastic': # random state
            # assert that len of actions is 1
            # assert len(actions) == 1

            # assert that actions are not in the prize_cards and between 1 and num_cards
            assert actions not in prize_cards and actions in range(1, num_cards+1)

            # append actions to prize_cards
            prize_cards = list(prize_cards)
            prize_cards.append(actions)
            prize_cards = tuple(prize_cards)

            # change state_type to simultaneous
            state_type = 'simultaneous'

        else:
            raise ValueError('Invalid state type: ' + state_type)
        
        return GOPSState(state_type, prize_cards, player_cards, opponent_cards, num_cards)

class GOPSActionEnumerator(ActionEnumerator):

    def __init__(self):
        super().__init__()

    def enumerate(self, state: GOPSState):
        '''
        Enumerates the possible actions that the player can take given the current state

        Args:
            state: current state

        Returns:
            actions: list of actions
        '''
        prize_cards = state.prize_cards
        player_cards = state.player_cards
        opponent_cards = state.opponent_cards
        num_cards = state.num_cards
        state_type = state.state_type

        starting_deck = list(range(1, num_cards+1))
        actions = list(set(starting_deck) - set(player_cards))
        return actions
    
class GOPSOpponentActionEnumerator(ActionEnumerator):

    def __init__(self):
        super().__init__()

    def enumerate(self, state: GOPSState, player = 0):
        '''
        Enumerates the possible actions that the opponent can take given the current state

        Args:
            state: current state

        Returns:
            actions: lists or set of actions
        '''
        prize_cards = state.prize_cards
        player_cards = state.player_cards
        opponent_cards = state.opponent_cards
        num_cards = state.num_cards
        state_type = state.state_type

        starting_deck = list(range(1, num_cards+1))
        actions = set(starting_deck) - set(opponent_cards)
        return actions
    
class GOPSRandomStateEnumerator(RandomStateEnumerator):

    def __init__(self):
        super().__init__()

    def enumerate(self, state: GOPSState):
        '''
        Enumerates the possible actions (prize card revealed) given the current state

        Args:
            state: current state

        Returns:
            actions: list of actions (cards)
        '''
        prize_cards = state.prize_cards
        num_cards = state.num_cards

        starting_deck = list(range(1, num_cards+1))
        actions = list(set(starting_deck) - set(prize_cards))
        return actions
    
class GOPSRandomStatePredictor(RandomStatePredictor):

    def __init__(self):
        super().__init__()

    def predict(self, state: GOPSState, actions):
        '''
        Predicts the probabilities over actions given the current state

        Args:
            state: current state
            actions: list of actions

        Returns:
            probs: dictionary of probabilities over actions
        '''
        probs = dict()
        for action in actions:
            probs[action] = 1.0/len(actions)
        return probs
        
class GPT35OpponentActionPredictor(OpponentActionPredictor):
    '''
    Opponent action predictor for GPT-3.5
    '''

    def __init__(self, model):
        '''
        Args:
            model: GPT-3.5 model
        '''
        self.model = model

    def predict(self, state: GOPSState, actions, player=0, prob=True) -> Dict:
        '''
        Predicts the advantage of each opponent action given the current state and action

        Args:
            state: current state
            actions: set or list of actions

        Returns:
            advantage: list of relative advantages of each opponent action (probs for current implementation)
        '''
        # Prepare input
        # print(actions)
        input_prompt = "Current State: {state}\nActions to take: {actions}\n".format(state=state.notes, actions=actions)
        # print(input_prompt)
        # print(type(input_prompt))
        # print(type(OPPONENT_ACTION_PREDICTOR_PROMPT))
        input_prompt += OPPONENT_ACTION_PREDICTOR_PROMPT
        # TODO: fix OPPONENT_ACTION_PREDICTOR_PROMPT to be better

        # Uncomment the following to use the model

        # # Call the model
        # output = self.model.single_action(input_prompt)

        # # Parse the output
        # advantages = parse_dict_with_any_key(output)

        import random
        advantages = {}
        print("Actions: ", actions)
        for action in actions:
            print("Action: ", action)
            advantages[action] = (1.0 * random.randint(0, len(actions)))/len(actions)

        # print(advantages)

        return advantages
    
class GPT35ValueHeuristic(ValueHeuristic):
    '''
    Value heuristic for GPT-3.5
    '''

    def __init__(self, model):
        '''
        Args:
            model: GPT-3.5 model
        '''
        self.model = model

    def evaluate(self, state: GOPSState) -> Dict:
        '''
        Predicts the value of the state

        Args:
            state: current state

        Returns:
            value: value of the state
        '''
        # Prepare input
        prob_prompt = "Current State: {state}\n".format(state=state.notes)
        prob_prompt += VALUE_PREDICTOR_PROMPTS[0]
        value_prompt = "Current State: {state}\n".format(state=state.notes)
        value_prompt += VALUE_PREDICTOR_PROMPTS[1]

        # Uncomment the following to use the model

        # # Call the model
        # prob_output = self.model.single_action(prob_prompt)
        # value_output = self.model.single_action(value_prompt)

        # # Parse the output
        # prob_value = parse_prob_value(prob_output)
        # value = parse_int_value(value_output)
        import numpy as np
        value = np.random.randint(0, 10)

        print(f"State: {state} Value: {value}")

        return value