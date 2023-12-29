import re
from typing import Dict, List
from headers import *
from prompts import *

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

    ((1,2,5), (2,3), (4,1), 6, 1)

    1,2,5 are the prize cards shown to the player in that order
    2,3 are the cards that the player has played in that order
    4,1 are the cards that the opponent has played in that order
    6 is the total number of cards in each deck or hand
    0,1,2 for the state type. 0 for max, 1 for min, 2 for random
    '''

    def __init__(self, state_type, prize_cards, player_cards, opponent_cards, num_cards):
        self.prize_cards = tuple(prize_cards)
        self.player_cards = tuple(player_cards)
        self.opponent_cards = tuple(opponent_cards)
        self.num_cards = num_cards
        self.state_type = state_type

        id = tuple([self.prize_cards, self.player_cards, self.opponent_cards, self.num_cards, state_type])
        turn = self.STATE_TYPES[state_type]
        super().__init__(id, turn)
        


class GOPSForwardEnumerator(ForwardEnumerator):

    def __init__(self):
        super().__init__()

    def enumerate(self, state: State, action):
        '''
        Enumerates the possible next states given the current state and action

        Args:
            state: current state
            action: action to take, which should be an integer indicating which card to play

        Returns:
            next_states: set of next states
        '''
        prize_cards = state.prize_cards
        player_cards = state.player_cards
        opponent_cards = state.opponent_cards
        num_cards = state.num_cards
        state_type = state.state_type

        if state_type == 0:
            # append action to player cards
            player_cards = list(player_cards)
            player_cards.append(action)
            player_cards = tuple(player_cards)

        elif state_type == 1:
            # append action to opponent cards
            opponent_cards = list(opponent_cards)
            opponent_cards.append(action)
            opponent_cards = tuple(opponent_cards)

        elif state_type == 2:
            # append action to prize cards
            prize_cards = list(prize_cards)
            prize_cards.append(action)
            prize_cards = tuple(prize_cards)

        # create next state
        next_state = GOPSState((state_type+1)%3, prize_cards, player_cards, opponent_cards, num_cards)

        return set([next_state])
    
class GOPSForwardPredictor(ForwardPredictor):

    def __init__(self):
        super().__init__()

    def predict(self, state: State, action, next_states):
        '''
        Predicts the probabilities over next states given the current state and action

        Args:
            state: current state
            action: action to take, which should be an integer indicating which card to play
            next_states: set of next states

        Returns:
            probs: dictionary of probabilities over next states
        '''
        probs = dict()
        probs[next_states[0]] = 1.0
        return probs

class GOPSActionEnumerator(ActionEnumerator):

    def __init__(self):
        super().__init__()

    def enumerate(self, state: State):
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

    def enumerate(self, state: State):
        '''
        Enumerates the possible actions that the opponent can take given the current state

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
        actions = list(set(starting_deck) - set(opponent_cards))
        return actions
    
class GOPSRandomStateEnumerator(RandomStateEnumerator):

    def __init__(self):
        super().__init__()

    def enumerate(self, state: State):
        '''
        Enumerates the possible next states given the current state

        Args:
            state: current state

        Returns:
            next_states: set of next states
        '''
        prize_cards = state.prize_cards
        player_cards = state.player_cards
        opponent_cards = state.opponent_cards
        num_cards = state.num_cards
        state_type = state.state_type

        starting_deck = list(range(1, num_cards+1))
        actions = list(set(starting_deck) - set(prize_cards))

        next_states = set()

        for action in actions:
            prize_cards = list(prize_cards)
            prize_cards.append(action)
            prize_cards = tuple(prize_cards)
            next_state = GOPSState((state_type+1)%3, prize_cards, player_cards, opponent_cards, num_cards)
            next_states.add(next_state)

        return next_states
    
class GOPSRandomStatePredictor(RandomStatePredictor):

    def __init__(self):
        super().__init__()

    def predict(self, state: State, next_states):
        '''
        Predicts the probabilities over next states given the current state

        Args:
            state: current state
            next_states: set of next states

        Returns:
            probs: dictionary of probabilities over next states
        '''
        probs = dict()
        for next_state in next_states:
            probs[next_state] = 1.0/len(next_states)
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

    def predict(self, state: State, actions) -> Dict:
        '''
        Predicts the advantage of each opponent action given the current state and action

        Args:
            state: current state
            actions: actions to take

        Returns:
            advantage: list of relative advantages of each opponent action (probs for current implementation)
        '''
        # Prepare input
        input_prompt = "Current State: {state}\nActions to take: {actions}\n".format(state=state.notes, actions=actions)
        input_prompt += OPPONENT_ACTION_PREDICTOR_PROMPT

        # Call the model
        output = self.model.single_action(input_prompt)

        # Parse the output
        advantages = parse_dict_with_any_key(output)

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

    def evaluate(self, state: State) -> Dict:
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

        # Call the model
        prob_output = self.model.single_action(prob_prompt)
        value_output = self.model.single_action(value_prompt)

        # Parse the output
        prob_value = parse_prob_value(prob_output)
        value = parse_int_value(value_output)

        return value