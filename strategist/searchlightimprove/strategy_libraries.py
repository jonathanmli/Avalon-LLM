from typing import Any, Hashable
from .headers import *
from searchlight.utils import UpdatablePriorityDictionary
from searchlight.datastructures.graphs import ValueGraph, StateTemplate

import numpy as np

class BFSStrategyLibrary(StrategyLibrary):
    '''
    Strategy library for using BFS to retrieve top strategies
    '''

    strategies_dict: UpdatablePriorityDictionary # where values are (abstract, feedback, iteration)

    def __init__(self) -> None:
        super().__init__()
        self.strategies_dict = UpdatablePriorityDictionary()

    def add_or_update_strategy(self, strategy: Hashable, notes: dict, score: float):
        '''
        Adds or updates a strategy in the strategies dictionary

        Args:
            strategy: strategy to add or update
            notes: notes for the strategy
            score: score of the strategy
        '''
        self.strategies_dict.add_or_update_key(strategy, notes, score)

    def add_seed_strategy(self, strategy: Any, notes: dict, score: float):
        return self.add_or_update_strategy(strategy, notes, score)

    def select_strategies(self, k: int = 1) -> list[tuple[Hashable, dict, float]]:
        '''
        Returns the k fittest items (highest to lowest). If there are less than k strategies, return all strategies

        Items of the form (strategy, info, score)
        '''
        return self.strategies_dict.get_top_k_items(k)
    
    def get_best_strategy(self) -> tuple[Hashable, dict, float]:
        '''
        Returns the best strategy

        Returns:
            best_strategy: best strategy
            best_info: best info
            best_score: best score
        '''
        return self.strategies_dict.get_top_k_items(1)[0]
    
class StrategyState(StateTemplate):
    score: float
    info: dict

    def __init__(self, strategy: Hashable, info: dict, score: float):
        super().__init__(strategy)
        self.info = info    
        self.score = score

    def copy(self):
        return StrategyState(self.id, dict(), self.score)
    
    @property
    def strategy(self) -> str:
        return self.id
    
    def __str__(self) -> str:
        '''
        Returns a string representation of the state as the hash of the state
        '''
        if isinstance(self.id, int):
            return f"StrategyState({self.id})"
        else:
            return f"StrategyState({hash(self)})"
    
    def __repr__(self):
        return self.__str__()


class MCTSStrategyLibrary(StrategyLibrary):
    '''
    Strategy library for using MCTS to retrieve top strategies

    Requires strategies to have (normalized) scores in [0, 1]
    '''

    strategy_graph: ValueGraph # where states are the strategies, edges are any improvement info, and intermediate rewards are the score improvement. 

    def __init__(self, rng: np.random.Generator = np.random.default_rng(), root_strategies: list[tuple[str, dict, float]] = []) -> None:
        super().__init__()
        self.strategy_graph = ValueGraph(players={0}, rng=rng)
        self.rng = rng
        self.root_states = []
        self.source_state = StrategyState(0, dict(), -1.0) # set to -1.0 for now so that it is lower than any strategy

        # add source state to graph
        self.add_state_to_graph(self.source_state)

        # create and add node for each root strategy
        for i, (strategy, info, score) in enumerate(root_strategies):
            self.add_seed_strategy(strategy, info, score)
            # self.root_states.append(root_state)

            # # add edge from source to root state
            # self.strategy_graph.add_child_to_state(self.source_state, root_state, f"seed_{i}", {0:score})

    def add_seed_state(self, state: StrategyState):
        '''
        Adds a seed state to the graph (by attaching it to the source state)
        '''
        self.logger.debug(f"Adding seed state {state}")
        self.add_state_to_graph(state)
        self.strategy_graph.add_child_to_state(self.source_state, state, f"seed_{len(self.root_states)}", {0:state.score})
        self.root_states.append(state)
    
    def add_seed_strategy(self, strategy: Any, notes: dict, score: float):
        '''
        Adds a seed strategy to the graph (by attaching it to the source state)
        '''
        state = StrategyState(strategy, notes, score)
        self.add_seed_state(state)

    def get_root_state(self) -> StrategyState:
        # if len(self.root_states) == 0:
        #     raise ValueError("No root states in the graph")
        # select a random root strategy using self.rng
        # return self.rng.choice(self.root_states)

        # return source state
        return self.source_state
    
    def add_state_to_graph(self, state: StrategyState):
        '''
        Adds a state to the graph
        '''
        try:
            node = self.strategy_graph.add_state(state)
            node.set_actor(0)
        except:
            self.logger.debug(f"State {state} already exists in the graph")


    def add_or_update_strategy(self, strategy: Hashable, notes: dict, score: float):
        '''
        Adds or updates a strategy in the strategies dictionary

        Args:
            strategy: strategy to add or update
            notes: notes for the strategy
            score: score of the strategy
        '''
        traj: list[tuple[Hashable, StrategyState]] = list(notes["last_trajectory"]) # NOTE: this should NOT contain the strategy to add or update
        # action = notes["last_idea"] # NOTE: last_idea does not exist for tree search
        state = StrategyState(strategy, notes, score)
        # see if state is in graph
        state_node = self.strategy_graph.get_node(state)
        if state_node is None:
            self.add_state_to_graph(state)
        else:
            state_node.state = state

        # determine the intermediate reward first. it should be the score if traj has length 1. otherwise should be difference between the last two scores
        if len(traj) <= 1: # means this is a seed state
            # this should never happen since trajectory shoud always include (1) source state and (2) seed state
            raise ValueError(f"Trajectory {traj} should always include source state and seed state")
            intermediate_reward = score
            self.add_seed_state(state)
        else:
            prev_score = traj[-1][1].score
            intermediate_reward = score - prev_score
            # add edge from last state in traj to state
            last_state = traj[-1][1]

            if last_state == state:
                # raise ValueError(f"Last state {last_state} is the same as state {state}")
                self.logger.debug(f"Last state {last_state} is the same as state {state}")
            else:
                self.logger.debug(f"Adding edge from {last_state} to {state} with action {hash(state)} and intermediate reward {intermediate_reward}")
                assert last_state.id != state.id
                self.strategy_graph.add_child_to_state(last_state, state, hash(state), {0:intermediate_reward})
                # traj[-1] = (last_state, action)
                
                # add state to traj
                traj.append((hash(state), state))

            self.logger.debug(f"Backproping trajectory {traj}")
            # now backpropagate the entire trajectory
            self.strategy_graph.backpropagate_trajectory(traj) # type: ignore

    def select_strategies(self, k: int = 1) -> list[tuple[Any, dict, float]]:
        '''
        Returns the k fittest items (highest to lowest). If there are less than k strategies, return all strategies

        Items of the form (strategy, info, score)
        '''
        if k == -1:
            # return all strategies in the graph
            all_nodes = self.strategy_graph.get_all_nodes()
            all_states: list[StrategyState] = [node.state for node in all_nodes]
            selected_items = [(state.strategy, state.info, state.score) for state in all_states]
            return selected_items
        else:
            selected_items = []
            # mcts simulate k times to get the top k strategies
            for i in range(k):
                root_state = self.get_root_state()
                traj: list[tuple[Hashable, StrategyState]] = self.strategy_graph.simulate_trajectory(state=root_state, stop_condition="negative_intermediate_reward") # type: ignore
                # get last state in trajectory
                last_state = traj[-1][1]
                strategy = last_state.strategy
                info = last_state.info
                info["last_trajectory"] = traj
                score = last_state.score
                selected_items.append((strategy, info, score))

            # self.logger.info(f"Selected strategies: {selected_items}")
            return selected_items
    
    def get_best_strategy(self) -> tuple[Any, dict, float]:
        '''
        Returns the best strategy

        Returns:
            best_strategy: best strategy
            best_info: best info
            best_score: best score
        '''
        # return the strategy with the highest info["score"]
        all_nodes = self.strategy_graph.get_all_nodes()
        # select all states except the source state
        all_states: list[StrategyState] = [node.state for node in all_nodes if not node.state == self.source_state]
        best_state = max(all_states, key=lambda x: x.score)
        return (best_state.strategy, best_state.info, best_state.score)