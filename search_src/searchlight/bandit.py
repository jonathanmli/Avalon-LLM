from .utils import UPDwithSampling, AbstractLogged
import numpy as np
from .datastructures.adjusters import UCBAdjuster, QValueAdjuster
from .datastructures.estimators import UtilityEstimatorMean, UtilityEstimator
from typing import Optional, Any
from collections import defaultdict
import logging

class MultiarmedBanditLearner(AbstractLogged):
    '''
    Lightweight class for a multiarmed bandit learner that supports the following operations:
    - add_or_update: update the reward of an arm
    - get_top_k_arms: get the top k arms with the highest estimated reward
    - softmax_sample: sample an arm using the softmax distribution
    '''

    arm_to_value_estimates: dict[Any, list[float]]

    def __init__(self, adjuster: QValueAdjuster = UCBAdjuster(), 
                 estimator: UtilityEstimator = UtilityEstimatorMean(),
                 rng: np.random.Generator = np.random.default_rng()):
        self.upd = UPDwithSampling(rng)
        self.adjuster = adjuster
        self.estimator = estimator
        self.arm_to_value_estimates = defaultdict(list)
        self.total_visits = 0
        
        super().__init__()

    def add_or_update(self, arm: Any, reward: float, notes: Any):
        '''
        Adds or updates the reward of an arm

        Args:
            arm: arm to update
            reward: reward to update
        '''
        self.arm_to_value_estimates[arm].append(reward)
        
        # compute value estimate on arm
        value_estimate = self.estimator.estimate_list(self.arm_to_value_estimates[arm])

        # compute the adjusted qvalue of the arm
        qvalue = self.adjuster.adjust(value_estimate, prior=1.0, 
                                      state_visits=self.total_visits, 
                                      state_action_visits=self.get_arm_visits(arm))
        
        # update the arm in the priority dictionary with the new qvalue
        self.upd.add_or_update_key(arm, notes, qvalue)
        self.logger.debug(f'Updated arm {arm} with reward {reward} and qvalue {qvalue}')
        self.total_visits += 1

    def get_notes_for_arm(self, arm: Any) -> Any:
        '''
        Returns the notes associated with an arm

        Args:
            arm: arm to check

        Returns:
            notes associated with the arm
        '''
        return self.upd.get_value(arm)

    def get_top_k_arms(self, k: int) -> list:
        '''
        Returns the top k arms with the highest estimated reward

        Args:
            k: number of arms to return

        Returns:
            list of top k arms
        '''
        items = self.upd.get_top_k_items(k)
        arms = [item[0] for item in items]
        return arms
    
    def get_top_k_items(self, k:int) -> list:
        '''
        Returns the top k items with the highest estimated reward

        Args:
            k: number of items to return

        Returns:
            list of top k items
        '''
        return self.upd.get_top_k_items(k)
    
    def softmax_sample(self, k: int = -1, temperature: float = 1.0) -> tuple[Any, Any, float]:
        '''
        Samples an arm using the softmax distribution

        Args:
            k: sample from the top k arms
            temperature: temperature parameter for the softmax distribution

        Returns:
            sampled item
        '''
        item = self.upd.softmax_sample(k, temperature)
        return item

    def get_arm_visits(self, arm: Any) -> int:
        '''
        Returns the number of visits to an arm

        Args:
            arm: arm to check

        Returns:
            number of visits to the arm
        '''
        return len(self.arm_to_value_estimates[arm])