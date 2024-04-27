from searchlight.utils import AbstractLogged
from ..Avalon.baseline_models_Avalon import AvalonState
from searchlight.gameplay.agents import SearchAgent
from searchlight.datastructures.graphs import ValueGraph2
from searchlight.datastructures.adjusters import PUCTAdjuster
from searchlight.algorithms.mcts_search import SMMonteCarlo
from searchlight.datastructures.estimators import UtilityEstimatorLast
import numpy as np

class ActionPlanner(AbstractLogged):

    def __init__(self, players, rng: np.random.Generator = np.random.default_rng()):
        # create value graph
        self.graph = ValueGraph2(adjuster=PUCTAdjuster(), utility_estimator=UtilityEstimatorLast(), rng=rng, players=players)

        # create search
        self.search = SMMonteCarlo(num_rollouts=100, rng=rng)

    def get_intent(self, state: AvalonState) -> str:
        raise NotImplementedError