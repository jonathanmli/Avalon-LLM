from search_src.searchlight.utils import AbstractLogged
from ..Avalon.baseline_models_Avalon import AvalonState
from search_src.searchlight.gameplay.agents import SearchAgent
from search_src.searchlight.datastructures.graphs import ValueGraph2
from search_src.searchlight.datastructures.adjusters import PUCTAdjuster
from search_src.searchlight.algorithms.mcts_search import SMMonteCarlo
from search_src.searchlight.datastructures.estimators import UtilityEstimatorMean
import numpy as np
from search_src.Avalon.baseline_models_Avalon import *

import numpy as np

def metropolis_hastings(n, k, p, n_iterations, rng: np.random.Generator = np.random.default_rng()):
    if len(p) != n:
        raise ValueError("The length of probability vector p must match n.")
    if k > n:
        raise ValueError("k must not be greater than n.")

    # Randomly initialize the state to satisfy the constraint of exactly k 1s
    state = np.zeros(n, dtype=int)
    ones_indices = rng.choice(n, k, replace=False)
    state[ones_indices] = 1
    
    # samples = []
    
    for _ in range(n_iterations):
        # Propose swapping two elements, one 1 with one 0 to maintain the constraint
        ones = np.where(state == 1)[0]
        zeros = np.where(state == 0)[0]
        swap_one = rng.choice(ones)
        swap_zero = rng.choice(zeros)
        
        # Create new state with swapped values
        new_state = state.copy()
        new_state[swap_one], new_state[swap_zero] = new_state[swap_zero], new_state[swap_one]
        
        # Calculate acceptance probability
        current_prob = np.prod([p[i] if state[i] == 1 else (1 - p[i]) for i in range(n)])
        new_prob = np.prod([p[i] if new_state[i] == 1 else (1 - p[i]) for i in range(n)])
        acceptance_prob = min(1, new_prob / current_prob)
        
        # Accept or reject the new state
        if rng.random() < acceptance_prob:
            state = new_state
        
        # samples.append(state.copy())
    
    return state


class ActionPlanner(AbstractLogged):

    def __init__(self, config: AvalonBasicConfig, func_str: str, player, player_role, known_sides, rng: np.random.Generator = np.random.default_rng(), num_rollouts=32):
        
        
        # create new game environment for simulation so that we don't mess up the main game environment
        env = AvalonGameEnvironment.from_num_players(config.num_players)
        num_players = env.config.num_players
        action_enumerator = AvalonActionEnumerator(env)
        forward_transitor = AvalonTransitor(env)
        actor_enumerator = AvalonActorEnumerator()
        policy_predictor = PolicyPredictor()
        start_state = AvalonState.init_from_env(env)
        players = set([i for i in range(num_players)])
        value_heuristic = AvalonLLMFunctionalValueHeuristic(func_str)


        # create initial inferencer
        initial_inferencer = PackageInitialInferencer(forward_transitor, action_enumerator, 
                                                 policy_predictor, actor_enumerator, 
                                                 value_heuristic)
        
        # create value graph
        self.graph = ValueGraph2(adjuster=PUCTAdjuster(), utility_estimator=UtilityEstimatorMean(), rng=rng, players=players)

        # create search
        self.search = SMMonteCarlo(initial_inferencer=initial_inferencer, rng=rng, node_budget=num_rollouts, num_rollout=1)

        self.agent = SearchAgent(search=self.search, graph=self.graph, rng=rng, player=player)
        self.action_enumerator = action_enumerator
        
        self.config = config
        self.role = player_role
        self.known_sides = known_sides
        self.player = player
        self.rng = rng
        self.num_rollouts = num_rollouts


    def get_intent(self, belief_p_is_good, belief_p_is_merlin, quest_leader, phase, turn, round, done, quest_team, team_votes, quest_results, good_victory) -> str:
        
        
        
        # we need to use our search algorithm to expand self.num_rollouts times
        for i in range(self.num_rollouts):
            # we need to sample from states here given the information set
            sampled_roles = self.sample_roles(belief_p_is_good, belief_p_is_merlin)

            # NOTE: quest votes is hidden information. given such, we will simply assume that good players always vote success and evil players always vote fail
            # hence, gather the votes for the players on the quest team
            quest_votes = tuple([AvalonBasicConfig.ROLES_TO_IS_GOOD[sampled_roles[player]] for player in quest_team])
            
            # we need to create a state from the sampled roles
            state = AvalonState(config=self.config, roles=sampled_roles, quest_leader=quest_leader, phase=phase, turn=turn, round=round, done=done, quest_team=quest_team, team_votes=team_votes, quest_votes=quest_votes, quest_results=quest_results, good_victory=good_victory)

            # we need to expand the graph
            self.search.expand(self.graph, state)

        # we need to get the best action from the agent
        legal_actions = self.action_enumerator.enumerate(state)
        return self.agent.act(state, legal_actions)
    
    def sample_roles(self, belief_p_is_good, belief_p_is_merlin,) -> tuple[int, ...]:
        # assert that both belief_p_is_good and belief_p_is_merlin are of the correct length
        assert len(belief_p_is_good) == self.config.num_players
        assert len(belief_p_is_merlin) == self.config.num_players

        # assert that belief_p_is_merlin is a probability vector
        assert np.all(belief_p_is_merlin >= 0)
        assert np.all(belief_p_is_merlin <= 1)
        assert np.isclose(np.sum(belief_p_is_merlin), 1)

        # assert that belief_p_is_good is a probability vector
        assert np.all(belief_p_is_good >= 0)
        assert np.all(belief_p_is_good <= 1)

        # if the player is a servant we need to sample the possible sides first
        if self.role == 5: # servant
            num_evil = self.config.num_evil
            num_players = self.config.num_players 
            # remove self from belief_p_is_good
            belief_p_is_good = np.delete(belief_p_is_good, self.player)
            # sample from the possible sides
            is_good = metropolis_hastings(num_players-1, num_evil, belief_p_is_good, 1000, self.rng)
            # insert player's own side
            is_good = np.insert(is_good, self.player, 1)
        else:
            # we assume that all sides are known to the player
            is_good = np.array(self.known_sides)
            # assert that there are no -1s in the known sides
            assert -1 not in is_good

        # next sample from the possible roles. recall that the player's role is known. We need to assign good roles to the good players and evil roles to the evil players

        # initialize the roles vector to all 5s
        roles = np.ones(self.config.num_players, dtype=int) * 5
        
        if self.role == 0:
            # Set the player's role to Merlin
            roles[self.player] = 0
        else:
            # Set a random good role to Merlin according to the belief
            p_is_merlin = np.where(is_good, belief_p_is_merlin, 0)
            p_is_merlin[self.player] = 0  # Player cannot be Merlin if they are not already

            # Normalize the vector
            p_is_merlin /= np.sum(p_is_merlin)

            # Sample from the vector
            merlin = self.rng.choice(self.config.num_players, p=p_is_merlin)
            roles[merlin] = 0


        # Set all is_good == 0 to Minion (6)
        roles[is_good == 0] = 6

        # Choose random evil player to be the Assassin (7) uniformly
        evil_players = np.where(roles == 6)[0]
        assassin = self.rng.choice(evil_players)
        roles[assassin] = 7

        return tuple(roles)