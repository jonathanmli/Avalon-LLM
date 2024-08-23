from .headers import *
from .datastructures.graphs import ValueGraph2

class ValueGraphBoltzmannPolicyPredictor(PolicyPredictor):

    def __init__(self, graph: ValueGraph2, adjusted_q = False):
        super().__init__()
        self.graph = graph
        self.adjusted_q = adjusted_q

    def _predict(self, state: State, actions, actor=None)-> dict:
        '''
        Predicts the policy probabilities across actions given the current state and actor

        Args:
            state: current state
            actions: list of actions
            actor: actor to predict policy for

        Returns:
            probs: dictionary of actions to probabilities
        '''
        node = self.graph.get_node(state)
        if node is None:
            # return uniform distribution if graph does not have the state
            return {action: 1/len(actions) for action in actions}
        else:
            # return the boltzmann distribution from the graph
            return self.graph.get_scheduled_boltzmann_policy(node=node, actor=actor, adjusted_q=self.adjusted_q)
