from ..datastructures.adjusters import Search
from ..headers import *
from ..datastructures.estimators import *
from ..datastructures.beliefs import *
from ..datastructures.adjusters import *
from ..datastructures.graphs import ValueGraph2
from ..algorithms.best_first_search import InferenceSearch2

class SMMonteCarlo(InferenceSearch2):
    '''
    Used to perform simultaneous expected monte carlo tree search 
    TODO: add alpha-beta pruning

    This only works if the opponent is a single agent and the game is a zero-sum game
    '''
    def __init__(self, initial_inferencer: InitialInferencer2,
                 rng: np.random.Generator = np.random.default_rng(),
                 num_rollout = 100, node_budget = 100, 
                 early_stopping_threshold: Optional[dict] = None,
                 cut_cycles: bool = False,):
        '''
        Args:
            initial_inferencer: initial inferencer to use
            rng: random number generator
            num_rollout: number of rollouts to run
            node_budget: number of nodes to expand
            early_stopping_threshold: threshold for early stopping (maps from player to threshold)
            cut_cycles: whether to cut cycles in the graph or not. if True, the graph will not add edges that create cycles. Otherwise, it will add edges that create cycles (leading to potentially infinite loops)
        '''
        super().__init__(initial_inferencer, cut_cycles=cut_cycles)
        self.rng = rng
        self.num_rollout = num_rollout
        self.node_budget = node_budget
        self.early_stopping_threshold = early_stopping_threshold
        
    def _expand(self, graph: ValueGraph2, state: State,):
        '''
        Expand starting from a node
        
        Args:
            state: state to expand from
            render: whether to render the graph or not
            revise: whether to revise the graph or not
            oracle: whether the opponent always plays the best response or not as if they knew the protagonist's policy
            alpha: alpha value for alpha-beta pruning
            beta: beta value for alpha-beta pruning
            threshold: threshold for alpha-beta pruning

        Returns:
            best_action: best action to take for player in the given state
        '''

        self.nodes_expanded = 0
        num_rollout = self.num_rollout

        # first expand belief tree by node_budget nodes
        while self.nodes_expanded < self.node_budget and num_rollout > 0:
            if self.early_stopping_threshold is not None:
                node = graph.get_node(state)
                if node is not None:
                    # for each player, check if the threshold is met
                    for player, threshold in self.early_stopping_threshold.items():
                        if graph.get_estimated_value(node, player) >= threshold:
                            return
            # run one simulation
            self.mc_simulate(graph, state)
            num_rollout -= 1

        # # select a joint action to take
        # node = graph.get_node(state)
        # joint_action = self.select_action(node, graph)

        # # get the action from joint action that corresponds to the player
        # best_action = dict(joint_action)[player]

        # return best_action

    def initial_inference(self, state: State):
        '''
        Conducts initial inference on a state
        '''
        # print('infering state', state)
        # print('initial_inferencer', self.initial_inferencer)
        return self.initial_inferencer.predict(state)
    
    def get_best_action(self, graph: ValueGraph2, state: State, actor=0):
        '''
        Get the best action to take for the given state
        '''
        node = graph.get_node(state)
        if node is None:
            return None
        return dict(graph.select_action(node))[actor]

    def mc_simulate(self, graph: ValueGraph2, state: State, prev_node = None, threshold = None) -> bool:
        '''
        Runs one simulation (rollout) from the given state

        The convention for joint actions is a tuple of tuples (actor, action), which can be converted to a dictionary
        using dict(joint_action)

        Args:
            graph: graph to simulate on
            state: state to simulate from
            prev_node: previous node in the graph
            threshold: threshold for early stopping (maps from player to threshold)

        Returns:
            is_expanded: whether the node is expanded or not
        '''
        # print('simulating state', state)

        node, did_expand = self.infer_node(graph, state)

        # check if node is terminal
        if node.get_actors() is None or not node.get_actors() or node.is_done():
            return False
        
        # if not then select an action to simulate
        action = graph.select_action(node)
        # print('selected action', action)
        # print('state', state)
        # print('possible actions', node.actions)

        # get the next state
        next_state = node.action_to_next_state[action]

        # simulate the next state
        is_expanded = self.mc_simulate(graph, next_state, node, threshold=threshold)

        # backpropagate the value from the next state
        graph.backpropagate(node, action, next_state)

        return is_expanded