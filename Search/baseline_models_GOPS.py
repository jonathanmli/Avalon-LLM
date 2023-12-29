from headers import *

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


        
