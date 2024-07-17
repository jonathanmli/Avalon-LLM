from search_src.searchlight.utils import AbstractLogged
from ..Avalon.baseline_models_Avalon import AvalonState
from search_src.searchlight.gameplay.agents import MuteMCTSAgent
# from search_src.searchlight.gameplay.agents import SearchAgent
from search_src.searchlight.datastructures.graphs import PartialValueGraph, PartialValueGraph2
from search_src.searchlight.datastructures.adjusters import PUCTAdjuster
from search_src.searchlight.datastructures.estimators import UtilityEstimatorMean
from .dialogue_discrimination import DialogueDiscriminator
from .dialogue_generator import DialogueGenerator
from .prompt_generator import PromptGenerator as PPromptGenerator
from search_src.searchlight.algorithms.mcts_search import SMMonteCarlo


import numpy as np
from collections import defaultdict
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


class AvalonActionPlannerAgent(MuteMCTSAgent):

    observed_dialogue: dict[tuple[int, int], list[tuple[int, str]]]
    role_to_dialogue_guide: dict[int, str]

    def __init__(self, config: AvalonBasicConfig, llm_model: LLMModel, player: int, value_heuristic: ValueHeuristic, role_to_dialogue_guide: dict[int, str], num_rollout: int = 100, node_budget: int = 100, rng: np.random.Generator = np.random.default_rng()):
        
        # create new game environment for simulation so that we don't mess up the main game environment
        env = AvalonGameEnvironment.from_num_players(config.num_players)
        num_players = env.config.num_players
        forward_transitor = AvalonTransitor(env)
        actor_action_enumerator = AvalonActorActionEnumerator(env)
        policy_predictor = PolicyPredictor()
        information_function = AvalonInformationFunction(config=config)
        
        # start_state = AvalonState.init_from_env(env)
        players = set([i for i in range(num_players)])
        self.config = config
        self.observed_dialogue = defaultdict(list)
        self.dialogue_gudies = role_to_dialogue_guide

        self.dialogue_discriminator = DialogueDiscriminator(llm_model=llm_model, prompt_generator=PPromptGenerator(self.config), player=player, players=players,)
        information_prior = AvalonInformationPrior(config=config, belief_p_is_merlin=self.dialogue_discriminator.get_p_is_merlin(), belief_p_is_good=self.dialogue_discriminator.get_p_is_good(),)

        self.dialogue_generator = DialogueGenerator(llm_model=llm_model, prompt_generator=PPromptGenerator(self.config),)

        super(). __init__(players=players, player=player, forward_transitor=forward_transitor, actor_action_enumerator=actor_action_enumerator, value_heuristic=value_heuristic, policy_predictor=policy_predictor, information_function=information_function, information_prior=information_prior, num_rollout=num_rollout, node_budget=node_budget, rng=rng)

    def _observe_dialogue(self, state: AvalonInformationSet, new_dialogue: list[tuple[int, str]]):
        '''
        Observes new dialogue and updates internal states
        '''

        if not self.dialogue_discriminator.check_role_equivalent(state.self_role):
            # reset the dialogue discriminator if the role is not equivalent
            self.dialogue_discriminator.reset(known_sides=state.known_sides, player_role=state.self_role)

        # we need to first convert the state and new_dialogue to string form here before passing it to the discriminator
        state_string_description = state.gen_str_description()
        new_dialogue_string = self.dialogue_list_to_str(new_dialogue)
        full_description = f"""{state_string_description}
        
        This round, players have said the following so far:
        {new_dialogue_string}"""

        # print("Full description: ", full_description)
        self.dialogue_discriminator.update_beliefs(full_description)
        self.observed_dialogue[(state.turn, state.round)].extend(new_dialogue)
        self.information_prior = AvalonInformationPrior(config=self.config, belief_p_is_merlin=self.dialogue_discriminator.get_p_is_merlin(), belief_p_is_good=self.dialogue_discriminator.get_p_is_good(),)# TODO: we need to update the information prior here
    
    @staticmethod
    def dialogue_list_to_str(dialogue: list[tuple[int, str]]):
        return '\n'.join([f"Player {player} said: {dialogue}" for player, dialogue in dialogue])

    def _produce_utterance(self, state: AvalonInformationSet,) -> str:
        '''
        Produces a dialogue given a history
        '''

        # get the dialogue for this turn and round
        dialogue = self.observed_dialogue[(state.turn, state.round)]

        # we need to first convert the state and new_dialogue to string form here before passing it to the discriminator
        state_string_description = state.gen_str_description()
        new_dialogue_string = self.dialogue_list_to_str(dialogue) # NOTE: might need to add preamble
        full_description = f"""{state_string_description}
        
        This round, players have said the following so far:
        {new_dialogue_string}"""

        # query action planner for intended action
        # actor, actions = self.actor_action_enumerator.enumerate(state)
        intended_action = self._act(state, actions=set("not yet decided")) # NOTE: we might need to change this to actions
        tips = self.dialogue_gudies[state.self_role]

        response, notes = self.dialogue_generator.generate_dialogue(history=full_description, phase=state.phase, intended_action=intended_action, tips=tips)
        return response

    # def _act(self, information_set: AvalonInformationSet, actions: set[Hashable]):
        

class BaselineAvalonActionPlannerAgent(AvalonActionPlannerAgent):
    '''
    Baseline agent for Avalon
    '''
    def __init__(self, config: AvalonBasicConfig, llm_model: LLMModel, player: int, value_heuristic: ValueHeuristic, role_to_dialogue_guide: dict[int, str], num_rollout: int = 100, node_budget: int = 100, rng: np.random.Generator = np.random.default_rng()):
        super().__init__(config=config, llm_model=llm_model, player=player, value_heuristic=value_heuristic, role_to_dialogue_guide=role_to_dialogue_guide, num_rollout=num_rollout, node_budget=node_budget, rng=rng)
        self.rng = rng
        env = AvalonGameEnvironment.from_num_players(config.num_players)
        num_players = env.config.num_players
        action_enumerator = AvalonActionEnumerator(env)
        forward_transitor = AvalonTransitor(env)
        actor_enumerator = AvalonActorEnumerator()
        policy_predictor = PolicyPredictor()
        players = set([i for i in range(num_players)])
        # create initial inferencer
        initial_inferencer = PackageInitialInferencer(forward_transitor, action_enumerator, 
                                                 policy_predictor, actor_enumerator, 
                                                 value_heuristic)
        
        # # create value graph
        self.graph = PartialValueGraph2(adjuster=PUCTAdjuster(), utility_estimator=UtilityEstimatorMean(), rng=rng, players=players)

        # create search
        self.search = SMMonteCarlo(initial_inferencer=initial_inferencer, rng=rng, node_budget=num_rollout, num_rollout=1)

    def _act(self, state: AvalonState, actions):
        # self.logger.info(f"Acting in state: {state}")
        belief_p_is_good = self.dialogue_discriminator.get_p_is_good()
        belief_p_is_merlin = self.dialogue_discriminator.get_p_is_merlin()
        (quest_leader, phase, turn, round, done, quest_team, team_votes, quest_results, known_sides, self_role) = state.get_information_set(self.player)

        # note that since our dialogue discriminator is initialized with the known sides and a role, we need to assert that the known sides and the role are the same as the ones in the state
        # self.logger.info(f"Player: {self.player}")
        # self.logger.info(f"Known sides: {known_sides}")
        # self.logger.info(f"self.sides: {self.known_sides}")
        # self.logger.info(f"Self role: {self_role}")
        # self.logger.info(f"Self.role: {self.role}")
        # print("self.known_sides: ", self.known_sides)
        # print("Self.role: ", self.role)
        self.known_sides = known_sides
        self.role = self_role
        # assert known_sides == self.known_sides
        # assert self_role == self.role
        # print("Calling act 2.1")

        # we need to use our search algorithm to expand self.num_rollouts times
        for i in range(self.num_rollout):
            # we need to sample from states here given the information set
            sampled_roles = self.sample_roles(belief_p_is_good, belief_p_is_merlin)
            # NOTE: quest votes is hidden information. given such, we will simply assume that good players always vote success and evil players always vote fail
            # hence, gather the votes for the players on the quest team
            quest_votes = tuple([AvalonBasicConfig.ROLES_TO_IS_GOOD[sampled_roles[player]] for player in quest_team])
            
            # we need to create a state from the sampled roles
            # new_state = AvalonState.init_from_state_tuple(config=self.config, roles=sampled_roles, quest_leader=quest_leader, phase=phase, turn=turn, round=round, done=done, quest_team=quest_team, team_votes=team_votes, quest_votes=quest_votes, quest_results=quest_results, good_victory=False)

            # we need to mc_simulate the graph for one rollout
            # self.search.expand(self.graph, new_state) # TODO: fix this
            self.search.mc_simulate(self.graph, state)
        # we need to get the best action from the agent
        # legal_actions = self.action_enumerator.enumerate(state, actor=self.player)
        # self.logger.info(f"Acting in state: {state}")
        return self.search.get_best_action(self.graph, state, self.player)
    

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