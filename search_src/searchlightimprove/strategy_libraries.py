from typing import Any
from .headers import *
from searchlight.utils import UpdatablePriorityDictionary
from searchlight.datastructures.graphs import ValueGraph, State

import numpy as np

class BFSStrategyLibrary(StrategyLibrary):
    '''
    Strategy library for using BFS to retrieve top strategies
    '''

    strategies_dict: UpdatablePriorityDictionary # where values are (abstract, feedback, iteration)

    def __init__(self) -> None:
        self.strategies_dict = UpdatablePriorityDictionary()

    def add_or_update_strategy(self, strategy: Any, notes: dict, score: float):
        '''
        Adds or updates a strategy in the strategies dictionary

        Args:
            strategy: strategy to add or update
            notes: notes for the strategy
            score: score of the strategy
        '''
        self.strategies_dict.add_or_update_key(strategy, notes, score)

    def select_strategies(self, k: int = 1) -> list[tuple[Any, dict, float]]:
        '''
        Returns the k fittest items (highest to lowest). If there are less than k strategies, return all strategies

        Items of the form (strategy, info, score)
        '''
        return self.strategies_dict.get_top_k_items(k)
    
    def get_best_strategy(self) -> tuple[Any, dict, float]:
        '''
        Returns the best strategy

        Returns:
            best_strategy: best strategy
            best_info: best info
            best_score: best score
        '''
        return self.strategies_dict.get_top_k_items(1)[0]
    
class StrategyState(State):
    score: float
    info: dict

    def __init__(self, strategy: str, info: dict, score: float):
        super().__init__(strategy)
        self.info = info    
        self.score = score

    def copy(self):
        return StrategyState(self.id, dict(), self.score)
    
    @property
    def strategy(self) -> str:
        return self.id


class MCTSStrategyLibrary(StrategyLibrary):
    '''
    Strategy library for using MCTS to retrieve top strategies
    '''

    strategy_graph: ValueGraph # where states are the strategies, edges are any improvement info, and intermediate rewards are the score improvement. 

    def __init__(self, rng: np.random.Generator = np.random.default_rng(), root_strategies: list[tuple[str, dict, float]] = []) -> None:
        self.strategy_graph = ValueGraph(players={0}, rng=rng)
        self.rng = rng
        self.root_states = []

        # create and add node for each root strategy
        for strategy, info, score in root_strategies:
            root_state = StrategyState(strategy, info, score)
            self.strategy_graph.add_state(root_state)
            self.root_states.append(root_state)

        

    def get_root_state(self) -> StrategyState:
        if len(self.root_states) == 0:
            raise ValueError("No root states in the graph")
        # select a random root strategy using self.rng
        return self.rng.choice(self.root_states)

    def add_or_update_strategy(self, strategy: Any, notes: dict, score: float):
        '''
        Adds or updates a strategy in the strategies dictionary

        Args:
            strategy: strategy to add or update
            notes: notes for the strategy
            score: score of the strategy
        '''
        traj: list[tuple[StrategyState, Any]] = list(notes["last_trajectory"]) # NOTE: this should NOT contain the strategy to add or update
        action = notes["last_idea"]
        state = StrategyState(strategy, notes, score)
        # see if state is in graph
        state_node = self.strategy_graph.get_node(state)
        if state_node is None:
            self.strategy_graph.add_state(state)
        else:
            state_node.state = state

        # determine the intermediate reward first. it should be the score if traj has length 1. otherwise should be difference between the last two scores
        if len(traj) == 0:
            intermediate_reward = score
            # add state to root states
            self.root_states.append(state)
        else:
            prev_score = traj[-1][0].score
            intermediate_reward = score - prev_score
            # add edge from last state in traj to state
            last_state = traj[-1][0]
            self.strategy_graph.add_child_to_state(last_state, state, action, {0:intermediate_reward})
            traj[-1] = (last_state, action)
            # add state to traj
            traj.append((state, None))
            # now backpropagate the entire trajectory
            self.strategy_graph.backpropagate_trajectory(traj)

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
                traj: list[tuple[StrategyState, Any]] = self.strategy_graph.simulate_trajectory(state=root_state, stop_condition="negative_intermediate_reward")
                # get last state in trajectory
                last_state = traj[-1][0]
                strategy = last_state.strategy
                info = last_state.info
                info["last_trajectory"] = traj
                score = last_state.score
                selected_items.append((strategy, info, score))

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
        all_states: list[StrategyState] = [node.state for node in all_nodes]
        best_state = max(all_states, key=lambda x: x.score)
        return (best_state.strategy, best_state.info, best_state.score)