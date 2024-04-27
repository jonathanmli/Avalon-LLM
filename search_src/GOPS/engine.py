import numpy as np

class GOPSConfig():
    '''
    GOPS Configuration
    '''
    def __init__(self, num_turns: int, custom_score_cards=None, random_state=None):
        '''
        num_turns: number of turns in a game
        custom_score_cards: custom score cards (should be of shape (num_turns,))
        '''
        self.num_turns = num_turns
        self.score_cards = custom_score_cards
        # playing cards should be numbered from 1 to num_turns
        self.playing_cards = np.arange(1, num_turns+1)
        # raise exception if custom_score_cards is not of shape (num_turns,)
        if self.score_cards is not None:
            assert self.score_cards.shape == (self.num_turns,)
        else:
            # default score cards are same as playing cards
            self.score_cards = self.playing_cards
        # set random state
        if random_state is None:
            random_state = np.random.RandomState()
        self.random_state = random_state

class GOPSEnvironment():
    '''
    GOPS Environment
    '''
    def __init__(self, config: GOPSConfig):
        '''
        config: GOPSConfig
        '''
        self.config = config
        self.reset()

    def reset(self):
        '''
        Resets the game state
        '''
        self.done = False
        self.current_turn = 0
        self.player1_hand = self.config.playing_cards.copy()
        self.player2_hand = self.config.playing_cards.copy()
        self.score_card_deck = self.config.score_cards.copy()
        self.player1_score = 0
        self.player2_score = 0
        self.contested_points = 0
        score_card = self._draw_score_card()
        return (self.done, score_card, self.contested_points)


    def _draw_score_card(self):
        '''
        Draws a score card
        '''
        # if score card deck is empty, return None
        if len(self.score_card_deck) == 0:
            return None
        else:
            score_card = self.config.random_state.choice(self.score_card_deck)
            self.score_card_deck = np.delete(self.score_card_deck, np.where(self.score_card_deck == score_card))
            self.contested_points += score_card
            return score_card
        
    def get_player1_hand(self):
        '''
        Returns player 1's hand
        '''
        return self.player1_hand
    
    def get_player2_hand(self):
        '''
        Returns player 2's hand
        '''
        return self.player2_hand
    
    def get_player1_score(self):
        '''
        Returns player 1's score
        '''
        return self.player1_score
    
    def get_player2_score(self):
        '''
        Returns player 2's score
        '''
        return self.player2_score
    
    def get_contested_points(self):
        '''
        Returns contested points
        '''
        return self.contested_points
    
    def get_current_turn(self):
        '''
        Returns current turn
        '''
        return self.current_turn
    
    def get_score_card_deck(self):
        '''
        Returns score card deck
        '''
        return self.score_card_deck
    
    def get_score_cards(self):
        '''
        Returns score cards
        '''
        return self.config.score_cards
    
    def get_num_turns(self):
        '''
        Returns number of turns
        '''
        return self.config.num_turns
    
    def play_cards(self, player1_card, player2_card):
        '''
        Plays the cards of the two players
        '''
        # check if cards are valid
        assert player1_card in self.player1_hand
        assert player2_card in self.player2_hand

        # remove cards from player hands
        self.player1_hand = np.delete(self.player1_hand, np.where(self.player1_hand == player1_card))
        self.player2_hand = np.delete(self.player2_hand, np.where(self.player2_hand == player2_card))

        # update score
        if player1_card > player2_card :
            self.player1_score += self.contested_points
            self.contested_points = 0
        elif player2_card  > player1_card:
            self.player2_score += self.contested_points
            self.contested_points = 0

        # update turn
        self.current_turn += 1

        # check if game is over
        if self.current_turn == self.config.num_turns:
            self.done = True
        
        # reveal random score card
        score_card = self._draw_score_card()

        # return done and score card
        return (self.done, score_card, self.contested_points)
    
