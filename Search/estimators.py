import numpy as np
from Search.beliefs import ValueNode

class UtilityEstimator:
    '''
    Abstract class for value estimators
    '''
    def __init__(self):
        pass

    def estimate(self, node: ValueNode):
        '''
        Estimates the value of a node

        Args:
            node: node to estimate

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

    def estimate(self, node: ValueNode):
        '''
        Estimates the value of a node

        Args:
            node: node to estimate

        Returns:
            utility: estimated utility of the node
        '''
        if len(node.values_estimates) == 0:
            return 0.0
        else:
            return np.mean(node.values_estimates)
        
class UtilityEstimatorLast(UtilityEstimator):
    '''
    Estimates the value of a node by taking the last value
    '''
    def __init__(self):
        super().__init__()

    def estimate(self, node: ValueNode):
        '''
        Estimates the value of a node

        Args:
            node: node to estimate

        Returns:
            utility: estimated utility of the node
        '''
        if len(node.values_estimates) == 0:
            return 0.0
        else:
            return node.values_estimates[-1]