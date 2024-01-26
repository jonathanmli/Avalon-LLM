from Search.beliefs import ValueGraph
from Search.headers import State
from Search.search import *
from Search.baseline_models_GOPS import *
from Search.engine import *
from Search.estimators import *
from Search.classic_models import *

class CustomBot:

    def __init__(self, player_id, random_state=None) -> None:
        self.player_id = player_id
        if random_state is None:
            random_state = np.random.RandomState()
        self.random_state = random_state

    def step(self, state):
        raise NotImplementedError

class SMMinimaxCustomBot(CustomBot):

    def __init__(self, id, hand, session, max_depth=3, num_rollouts=100, random_state: np.random.RandomState=None):
        """Initializes the SMMinimaxBot.
        
        Args:
            player_id: player id
            random_state: random state
            max_depth: maximum depth to search
            num_rollouts: number of rollouts to perform for value estimation
        """
        super().__init__(id, random_state)
        self.hand = hand
        self.num_cards = len(hand)
        if random_state is None:
            random_state = np.random.RandomState(42)
        self.max_depth = max_depth
        self.value_graph = ValueGraph()
        self.action_enumerator = GOPSActionEnumerator()
        self.action_predictor = GOPSRandomActionPredictor()
        self.forward_transitor = GOPSForwardTransitor()
        self.utility_estimator = UtilityEstimatorLast()
        self.actor_enumerator = GOPSActorEnumerator()
        self.value_heuristic = RandomRolloutValueHeuristic(self.actor_enumerator, self.action_enumerator, 
                                                      self.forward_transitor, num_rollouts=num_rollouts,
                                                      random_state=self.random_state)
        self.search = SMMinimax(self.forward_transitor, self.value_heuristic, self.actor_enumerator,
                                self.action_enumerator, self.action_predictor, 
                                self.utility_estimator)
        
    async def initialize(self):
        pass

    async def step(self, state: str=None, opponent_hand: List=None, contested_scores: int=None, score_card_left: List=None, **kwargs):
        """Returns the action to be taken by this bot in the given state."""
        prize_cards = kwargs.pop("prize_cards")
        player_cards = kwargs.pop("player_cards")
        opponent_cards = kwargs.pop("opponent_cards")
        num_cards = self.num_cards
        gops_state = GOPSState({0,1}, prize_cards, player_cards, opponent_cards, num_cards, False)
        print(gops_state)
        # then expand the value graph
        self.search.expand(self.value_graph, gops_state, depth=self.max_depth)

        # then get the best action from the value graph
        action = self.value_graph.get_best_action(gops_state)

        card = action

        self.hand = np.delete(np.array(self.hand), np.where(np.array(self.hand) == card)).tolist()

        return action