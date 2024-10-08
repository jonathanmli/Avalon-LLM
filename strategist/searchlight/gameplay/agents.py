from collections.abc import Hashable
from strategist.searchlight.headers import *
from strategist.searchlight.datastructures.graphs import *
from strategist.searchlight.datastructures.beliefs import *

class HumanAgent(Agent):
    '''
    Queries the user for actions
    '''

    def __init__(self, player: int, action_parser: Callable[[str], Hashable] = lambda x: x, rng: np.random.Generator = np.random.default_rng()):
        print("Welcome to the game, human. You are player", player)
        super().__init__(player)
        self.action_parser = action_parser

    def _act(self, state: Hashable, actions: set[Hashable]) -> Hashable:
        '''
        Queries the user for an action
        '''
        print("State:", state)
        print("Legal actions:", actions)
        while True:
            try:
                action = input(f"Player {self.player} enter action: ")
                action = self.action_parser(action)
                if action in actions:
                    break
                else:
                    print("Not legal action, try again")
            except Exception as e:
                print("Could not parse action, try again")
        return action

class MCTSAgent(Agent):

    def __init__(self, players: set[int], player: int, forward_transitor: ForwardTransitor, actor_action_enumerator: ActorActionEnumerator, value_heuristic: ValueHeuristic, policy_predictor: Optional[PolicyPredictor] = None, information_function: Optional[InformationFunction] = None, information_prior: Optional[InformationPrior] = None, num_rollout: int = 100, node_budget: int = 100, early_stopping_threshold = None, cut_cycles = False, rng: np.random.Generator = np.random.default_rng()):
        '''
        Creates an MCTS agent
        '''
        if information_function is None:
            self.graph = ValueGraph(players=players, adjuster="puct", utility_estimator="mean", rng=rng)
            self.information_function = None
        else:
            self.graph = PartialValueGraph(players=players, adjuster="puct", utility_estimator="mean", rng=rng)
            self.information_function = information_function
            self.information_prior = information_prior
            assert information_prior is not None, "Information prior must be provided if information function is provided"

        self.num_rollout = num_rollout
        self.node_budget = node_budget
        self.early_stopping_threshold = early_stopping_threshold
        self.cut_cycles = cut_cycles

        self.forward_transitor = forward_transitor
        self.actor_action_enumerator = actor_action_enumerator
        self.value_heuristic = value_heuristic
        self.policy_predictor = policy_predictor
        super().__init__(player)

    def is_partial_graph(self):
        return self.information_function is not None

    def _act(self, state: Hashable, actions: set[Hashable],) -> Hashable:
        # print("acting with state", state, "actions", actions)
        # print("class of state", state.__class__)
        # expand the graph using MCTS first
        print("actioning")
        self.expand(initial_state=state)
        print("expanded")
        
        if not self.is_partial_graph():
            print("not partial")
            # get the best action from graph
            action = self.graph.get_best_action(state=state, actions=actions)
        else:
            print("partial")
            # first assert that state is in the graph
            # assert self.graph.get_information_set_node(state) is not None, f"Information set {repr(state)} must be in the graph"
            # get the best action from graph
            action = self.graph.get_best_action_from_information_set(information_set=state, actions=actions) # type: ignore
        return action

    def expand(self, initial_state: Hashable):
        '''
        Expands the knowledge graph of the agent using MCTS simulation
        '''
        # print("expanding with state", initial_state)
        # print("class of state", initial_state.__class__)
        print(f"Initial State: {initial_state}")
        nodes_expanded = 0
        for _ in range(self.num_rollout):
            if nodes_expanded >= self.node_budget:
                break
            if not self.is_partial_graph():
                print("expand not partial")
                # simulate a trajectory from the state
                trajectory = self.graph.simulate_trajectory(state=initial_state, stop_condition="has_unvisited_actions")
                state = initial_state
            else:
                print("expand partial")
                state = self.graph.select_hidden_state(information_set=initial_state, information_prior=self.information_prior) # FIXME # type: ignore

                # print(f"Selected hidden state {state} from information set {initial_state}")

                trajectory = self.graph.simulate_trajectory(state=state, stop_condition="has_unvisited_actions")
                # print("simulated trajectory", trajectory)
                # trajectory = self.graph.simulate_trajectory_from_information_set(information_set=state, information_prior= self.information_prior, stop_condition="has_unvisited_actions")
            
            if len(trajectory) == 0:
                print("trajectory is empty")
                # this means we need to expand the state first
                # overwrite the next_state with initial inference information
                actor, actions = self.actor_action_enumerator.enumerate(state)
                actor_to_value_estimates, _ = self.value_heuristic.evaluate(state)
                if self.policy_predictor is not None:
                    action_to_prob_weights = self.policy_predictor.predict(state = state, actor = actor, actions = actions)
                else:
                    action_to_prob_weights = defaultdict(lambda: 1.0)
                notes = dict()

                if not self.is_partial_graph(): # full information or terminal state ## or actor is None
                    self.graph.overwrite_state(state=state, actor=actor, actions=actions, actor_to_value_estimates=actor_to_value_estimates, notes=notes, prob_weights=action_to_prob_weights)
                else:
                    print("simulated state", state)
                    information_set = self.information_function.get_information_set(state=state, actor=actor) # type: ignore
                    # assert information_set == initial_state, f"Simulated information set {repr(information_set)} must be equal to initial state {repr(initial_state)}" # NOTE: no need, we can simulate from different information set
                    self.graph.overwrite_state(state=state, actor=actor, actions=actions, actor_to_value_estimates=actor_to_value_estimates, notes=notes, prob_weights=action_to_prob_weights, information_set=information_set) # type: ignore
            else:
                # last state will either be terminal or have unvisited actions
                last_state = trajectory[-1][1]
                # attempt to pick a random action from unvisited actions
                selected_action = self.graph.select_random_unvisited_action(last_state)
                actor = self.graph.get_actor(last_state)
                if selected_action is None:
                    continue
                
                # use forward_transitor to get next state and intermediate rewards
                next_state, actor_to_reward = self.forward_transitor.transition(last_state, selected_action, actor)

                # add child to state
                # NOTE: this should be the only place where we add new nodes to the graph
                self.graph.add_child_to_state(state=last_state, action=selected_action, child_state=next_state, rewards=actor_to_reward)

                # overwrite the next_state with initial inference information
                actor, actions = self.actor_action_enumerator.enumerate(next_state)
                actor_to_value_estimates, _ = self.value_heuristic.evaluate(next_state)
                if self.policy_predictor is not None:
                    action_to_prob_weights = self.policy_predictor.predict(state = next_state, actor = actor, actions = actions)
                else:
                    action_to_prob_weights = defaultdict(lambda: 1.0)
                notes = dict()

                if self.information_function is None or actor is None: # full information or terminal state
                    self.graph.overwrite_state(state=next_state, actor=actor, actions=actions, actor_to_value_estimates=actor_to_value_estimates, notes=notes, prob_weights=action_to_prob_weights)
                else:
                    information_set = self.information_function.get_information_set(state=next_state, actor=actor)
                    self.graph.overwrite_state(state=next_state, actor=actor, actions=actions, actor_to_value_estimates=actor_to_value_estimates, notes=notes, prob_weights=action_to_prob_weights, information_set=information_set) # type: ignore

                # append next state and action to trajectory
                trajectory.append((selected_action, next_state))

                # backpropagate the trajectory
                self.graph.backpropagate_trajectory(trajectory)

            nodes_expanded += 1
        
        # should most definitely have the initial state in the graph
        # if self.is_partial_graph():
        #     assert self.graph.get_information_set_node(initial_state) is not None, f"Information set {repr(initial_state)} must be in the graph after expansion"
        # else:
        #     assert self.graph.get_node(initial_state) is not None, f"State {repr(initial_state)} must be in the graph after expansion"

class MuteMCTSAgent(MCTSAgent, DialogueAgent):
    '''
    MCTS agent that does not speak
    '''

    def _observe_dialogue(self, state: Hashable, new_dialogue: list[tuple[int, str]]):
        pass

    def _produce_utterance(self, state: Hashable) -> str:
        return ""

class HumanDialogueAgent(HumanAgent, DialogueAgent):
    '''
    Queries the user for actions
    '''
    def _observe_dialogue(self, state: Hashable, new_dialogue: list[tuple[int, str]]):
        # print("State:", state)
        print("New dialogue: \n", self.dialogue_list_to_str(new_dialogue))

    def _produce_utterance(self, state: Hashable) -> str:
        '''
        Queries the user for an utterance
        '''
        print("State:", state)
        return input(f"Player {self.player} enter utterance: ")

            

