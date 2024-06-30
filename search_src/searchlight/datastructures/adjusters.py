from ..headers import *
from collections import deque
import warnings
import itertools
from datetime import datetime
import numpy as np
from queue import PriorityQueue
    
class QValueAdjuster:
    '''
    Abstract class. Used to adjust qvalues for a given node
    '''
    def __init__(self) -> None:
        pass

    def adjust(self, qvalue: float, prior: float, state_visits: int, state_action_visits: int) -> float:
        '''
        Adjusts the qvalue for a given state-action pair

        Args:
            qvalue: qvalue to adjust
            prior: prior probability of the action
            state_visits: number of visits to the state
            state_action_visits: number of visits to the state-action pair

        Returns:
            adjusted qvalue
        '''
        return qvalue
    
class PUCTAdjuster(QValueAdjuster):

    def __init__(self, c1=1.0, c2=19652) -> None:
        self.c1 = c1
        self.c2 = c2

    def adjust(self, qvalue, prior, state_visits, state_action_visits) -> float:
        '''
        Adjusts the qvalue for a given state-action pair

        Args:
            qvalue: qvalue to adjust
            prior: prior probability of the action
            state_visits: number of visits to the state
            state_action_visits: number of visits to the state-action pair

        Returns:
            adjusted qvalue
        '''
        return qvalue + prior*np.sqrt(state_visits)/(1 + state_action_visits) *(self.c1 + np.log((state_visits + self.c2 + 1)/self.c2))
    
class UCBAdjuster(QValueAdjuster):

    def __init__(self, c=np.sqrt(2)) -> None:
        self.c = c
        self.small_value = 1e-2

    def adjust(self, qvalue: float, prior: float=1.0, state_visits: int=1, state_action_visits:int=1) -> float:
        '''
        Adjusts the qvalue for a given state-action pair

        Args:
            qvalue: qvalue to adjust
            prior: prior probability of the action
            state_visits: number of visits to the state
            state_action_visits: number of visits to the state-action pair

        Returns:
            adjusted qvalue
        '''
        # if state_action_visits == 0, set it to a small value
        state_action_visits = max(state_action_visits, self.small_value)
        # print('ucb', qvalue, prior, state_visits, state_action_visits, self.c)
        return qvalue + prior*np.sqrt(np.log(state_visits)/(state_action_visits)) *self.c

    