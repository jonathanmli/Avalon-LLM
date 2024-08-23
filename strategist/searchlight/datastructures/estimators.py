import numpy as np
from .beliefs import ValueNode

class UtilityEstimator:
    '''
    Abstract class for value estimators
    '''
    def __init__(self):
        pass

    def estimate(self, node: ValueNode, actor=0) -> float:
        '''
        Estimates the value of a node

        Args:
            node: node to estimate

        Returns:
            utility: estimated utility of the node
        '''
        raise NotImplementedError
    
    @staticmethod
    def estimate_list(value_estimates: list[float]):
        '''
        Estimates the value given a list of value estimates

        Args:
            value_estimates: list of value estimates

        Returns:
            utility: estimated utility of the node
        '''
        raise NotImplementedError

class UtilityEstimatorMean(UtilityEstimator):
    '''
    Estimates the value of a node by taking the mean of the values
    '''
    def __init__(self):
        super().__init__()

    def estimate(self, node: ValueNode, actor=None) -> float:
        '''
        Estimates the value of a node

        Args:
            node: node to estimate
            actor: (optional) actor to estimate the value for

        Returns:
            utility: estimated utility of the node
        '''
        # if actor is None:
        #     if len(node.values_estimates) == 0:
        #         return 0.0
        #     else:
        #         return np.mean(node.values_estimates)
        # else:
        if len(node.actor_to_value_estimates.get(actor, [])) == 0:
            return 0.0
        else:
            return np.mean(node.actor_to_value_estimates[actor])
    
    @staticmethod
    def estimate_list(value_estimates: list[float]):
        '''
        Estimates the value given a list of value estimates

        Args:
            value_estimates: list of value estimates

        Returns:
            utility: estimated utility of the node
        '''
        if len(value_estimates) == 0:
            return 0.0
        else:
            return np.mean(value_estimates)
        
class UtilityEstimatorLast(UtilityEstimator):
    '''
    Estimates the value of a node by taking the last value
    '''
    def __init__(self):
        super().__init__()

    def estimate(self, node: ValueNode, actor=None) -> float:
        '''
        Estimates the value of a node, optionally for a specific actor

        Args:
            node: node to estimate
            actor: (optional) actor to estimate the value for

        Returns:
            utility: estimated utility of the node
        '''
        if len(node.actor_to_value_estimates.get(actor, [])) == 0:
            return 0.0
        else:
            return node.actor_to_value_estimates[actor][-1]
    
    @staticmethod
    def estimate_list(value_estimates: list[float]):
        '''
        Estimates the value given a list of value estimates

        Args:
            value_estimates: list of value estimates

        Returns:
            utility: estimated utility of the node
        '''
        if len(value_estimates) == 0:
            # this usually happens when the node is terminal
            return 0.0
        else:
            return value_estimates[-1]

